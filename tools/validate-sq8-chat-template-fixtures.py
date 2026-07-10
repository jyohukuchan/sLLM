#!/usr/bin/env python3
"""Independently validate SQ8 Qwen3 chat-template fixtures against local files."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import operator
import os
import struct
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "ullm.sq8.chat_template_fixtures.v1"
FIXTURE_SCHEMA_VERSION = "ullm.sq8.chat_template_fixture.v1"
FIXTURE_SET_ID = "qwen3-14b-fp8-sq8-chat-template-v0.1"
MODEL_ID = "Qwen/Qwen3-14B-FP8"
EXPECTED_REVISION = "9a283b4a5efbc09ce247e0ae5b02b744739e525a"
DEFAULT_MODEL_DIR = Path(
    "/home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3-14B-FP8"
)
EXACT_PROMPT_LENGTHS = (32, 128, 512, 2_048, 3_584)
BASE_PROMPT_TOKENS = 12
REPEATED_UNIT = " x"
CONTEXT_LENGTH = 4_096
TOKENIZER_REQUIRED_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "generation_config.json",
)
TOKENIZER_OPTIONAL_FILES = (
    "added_tokens.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "spiece.model",
    "chat_template.jinja",
)


class ValidationError(RuntimeError):
    """Raised when fixture bytes or independently rendered values differ."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture_dir", type=Path)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    return parser.parse_args()


def absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, relative: str) -> dict[str, Any]:
    return {
        "file": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValidationError(f"non-finite JSON value: {value}")
            ),
        )
    except ValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"failed to parse {path}: {error}") from error


def require_exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise ValidationError(
            f"{label} member set mismatch: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )
    return value


def read_revision(model_dir: Path, filename: str) -> str:
    metadata = model_dir / ".cache" / "huggingface" / "download" / f"{filename}.metadata"
    try:
        lines = metadata.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValidationError(f"missing revision metadata for {filename}: {error}") from error
    if not lines or lines[0] != EXPECTED_REVISION:
        actual = lines[0] if lines else None
        raise ValidationError(
            f"revision mismatch for {filename}: {actual!r} != {EXPECTED_REVISION!r}"
        )
    return lines[0]


def snapshot_model_identity(model_dir: Path) -> dict[str, Any]:
    if not model_dir.is_dir():
        raise ValidationError(f"model directory does not exist: {model_dir}")
    config_path = model_dir / "config.json"
    config = load_json(config_path)
    if not isinstance(config, dict) or config.get("model_type") != "qwen3":
        raise ValidationError("model config is not the fixed Qwen3 model type")

    tokenizer_names = list(TOKENIZER_REQUIRED_FILES)
    for name in TOKENIZER_OPTIONAL_FILES:
        if (model_dir / name).is_file():
            tokenizer_names.append(name)

    tokenizer_files = []
    for name in tokenizer_names:
        path = model_dir / name
        if not path.is_file():
            raise ValidationError(f"missing tokenizer file: {path}")
        tokenizer_files.append(file_record(path, name))

    revision_files = ["config.json", *tokenizer_names]
    revisions = {name: read_revision(model_dir, name) for name in revision_files}
    return {
        "model": {
            "id": MODEL_ID,
            "revision": EXPECTED_REVISION,
            "model_type": "qwen3",
            "config": file_record(config_path, "config.json"),
        },
        "tokenizer_revision": EXPECTED_REVISION,
        "tokenizer_files": tokenizer_files,
        "revision_evidence": revisions,
    }


def load_local_tokenizer(model_dir: Path) -> tuple[Any, str]:
    try:
        import transformers
        from transformers import AutoTokenizer
    except ImportError as error:
        raise ValidationError(f"transformers is unavailable: {error}") from error
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as error:
        raise ValidationError(f"failed to load the local tokenizer: {error}") from error
    return tokenizer, str(transformers.__version__)


def chat_template_identity(tokenizer: Any) -> dict[str, Any]:
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template:
        raise ValidationError("tokenizer.chat_template must be a non-empty string")
    encoded = template.encode("utf-8")
    return {
        "utf8_bytes": len(encoded),
        "sha256": sha256_bytes(encoded),
    }


def extract_input_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise ValidationError("tokenized chat template has no input_ids")
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValidationError("tokenized chat template input_ids is not a sequence")
    if value and isinstance(value[0], Sequence):
        if len(value) != 1:
            raise ValidationError("tokenized chat template returned more than one sequence")
        value = value[0]

    result = []
    for index, item in enumerate(value):
        if isinstance(item, bool):
            raise ValidationError(f"token ID {index} is boolean")
        try:
            token_id = operator.index(item)
        except TypeError as error:
            raise ValidationError(f"token ID {index} is not an integer") from error
        if token_id < 0 or token_id > 0xFFFF_FFFF:
            raise ValidationError(f"token ID {index} is outside unsigned 32-bit range")
        result.append(int(token_id))
    if not result:
        raise ValidationError("chat template produced no token IDs")
    return result


def render_messages(tokenizer: Any, messages: list[dict[str, str]]) -> tuple[str, list[int]]:
    keyword_args = {
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    rendered = tokenizer.apply_chat_template(
        copy.deepcopy(messages),
        tokenize=False,
        **keyword_args,
    )
    if not isinstance(rendered, str):
        raise ValidationError("non-tokenized chat template output is not a string")
    encoded = tokenizer.apply_chat_template(
        copy.deepcopy(messages),
        tokenize=True,
        **keyword_args,
    )
    return rendered, extract_input_ids(encoded)


def expected_fixture_definitions() -> list[dict[str, Any]]:
    exact = []
    for target in EXACT_PROMPT_LENGTHS:
        repeat = target - BASE_PROMPT_TOKENS
        exact.append(
            {
                "fixture_id": f"exact-p{target:04d}",
                "kind": "exact_length",
                "messages": [{"role": "user", "content": REPEATED_UNIT * repeat}],
                "construction": {
                    "type": "single_user_repeated_unit",
                    "base_prompt_tokens": BASE_PROMPT_TOKENS,
                    "unit": REPEATED_UNIT,
                    "repeat": repeat,
                    "target_prompt_tokens": target,
                },
            }
        )
    representative = [
        {
            "fixture_id": "english-user",
            "kind": "representative",
            "messages": [
                {
                    "role": "user",
                    "content": "Explain why deterministic fixtures matter in one sentence.",
                }
            ],
            "construction": {"type": "representative", "category": "english"},
        },
        {
            "fixture_id": "japanese-user",
            "kind": "representative",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "決定的なテスト用データが必要な理由を一文で説明してください。"
                    ),
                }
            ],
            "construction": {"type": "representative", "category": "japanese"},
        },
        {
            "fixture_id": "system-user",
            "kind": "representative",
            "messages": [
                {"role": "system", "content": "Answer concisely and accurately."},
                {"role": "user", "content": "Name the capital of Japan."},
            ],
            "construction": {"type": "representative", "category": "system_user"},
        },
        {
            "fixture_id": "two-turn",
            "kind": "representative",
            "messages": [
                {"role": "user", "content": "What is two plus two?"},
                {"role": "assistant", "content": "Four."},
                {"role": "user", "content": "Multiply that result by three."},
            ],
            "construction": {"type": "representative", "category": "two_turn"},
        },
        {
            "fixture_id": "code-block",
            "kind": "representative",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Review this code:\n\n```python\ndef square(x):\n"
                        "    return x * x\n```"
                    ),
                }
            ],
            "construction": {"type": "representative", "category": "code_block"},
        },
    ]
    return [*exact, *representative]


def token_ids_bytes(token_ids: Sequence[int]) -> bytes:
    return b"".join(struct.pack("<I", token_id) for token_id in token_ids)


def validate_manifest_identity(
    manifest: dict[str, Any],
    identity: dict[str, Any],
    tokenizer: Any,
    transformers_version: str,
) -> None:
    require_exact_keys(
        manifest,
        {
            "schema_version",
            "fixture_set_id",
            "model",
            "tokenizer",
            "template_options",
            "exact_length_contract",
            "fixture_files",
        },
        "manifest",
    )
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValidationError("manifest schema_version mismatch")
    if manifest["fixture_set_id"] != FIXTURE_SET_ID:
        raise ValidationError("manifest fixture_set_id mismatch")
    if manifest["model"] != identity["model"]:
        raise ValidationError("manifest model identity differs from local files")

    tokenizer_record = require_exact_keys(
        manifest["tokenizer"],
        {
            "revision",
            "class",
            "transformers_version",
            "loader",
            "files",
            "revision_evidence",
            "chat_template",
        },
        "manifest.tokenizer",
    )
    if tokenizer_record["revision"] != identity["tokenizer_revision"]:
        raise ValidationError("manifest tokenizer revision mismatch")
    if tokenizer_record["class"] != type(tokenizer).__name__:
        raise ValidationError("manifest tokenizer class mismatch")
    if tokenizer_record["transformers_version"] != transformers_version:
        raise ValidationError("manifest transformers version differs from validator runtime")
    if tokenizer_record["loader"] != {
        "local_files_only": True,
        "trust_remote_code": False,
    }:
        raise ValidationError("manifest tokenizer loader flags are not fixed")
    if tokenizer_record["files"] != identity["tokenizer_files"]:
        raise ValidationError("manifest tokenizer file hashes differ from local files")
    if tokenizer_record["revision_evidence"] != identity["revision_evidence"]:
        raise ValidationError("manifest revision evidence differs from local metadata")
    if tokenizer_record["chat_template"] != chat_template_identity(tokenizer):
        raise ValidationError("manifest chat-template hash differs from local tokenizer")
    if manifest["template_options"] != {
        "add_generation_prompt": True,
        "enable_thinking": False,
    }:
        raise ValidationError("manifest template options are not fixed")
    if manifest["exact_length_contract"] != {
        "base_prompt_tokens": BASE_PROMPT_TOKENS,
        "message_shape": "single user",
        "content_rule": "' x' * repeat",
        "target_prompt_tokens": list(EXACT_PROMPT_LENGTHS),
    }:
        raise ValidationError("manifest exact-length contract mismatch")


def validate_fixture_payload(
    payload: Any,
    definition: dict[str, Any],
    rendered_text: str,
    token_ids: list[int],
) -> None:
    fixture_id = definition["fixture_id"]
    fixture = require_exact_keys(
        payload,
        {
            "schema_version",
            "fixture_id",
            "kind",
            "messages",
            "construction",
            "template_options",
            "expected",
        },
        f"fixture {fixture_id}",
    )
    if fixture["schema_version"] != FIXTURE_SCHEMA_VERSION:
        raise ValidationError(f"fixture {fixture_id} schema_version mismatch")
    for key in ("fixture_id", "kind", "messages", "construction"):
        if fixture[key] != definition[key]:
            raise ValidationError(f"fixture {fixture_id} {key} mismatch")
    if fixture["template_options"] != {
        "add_generation_prompt": True,
        "enable_thinking": False,
    }:
        raise ValidationError(f"fixture {fixture_id} template options mismatch")

    expected = require_exact_keys(
        fixture["expected"],
        {
            "prompt_tokens",
            "rendered_text",
            "rendered_text_utf8_bytes",
            "rendered_text_sha256",
            "token_ids",
            "token_ids_u32le_bytes",
            "token_ids_u32le_sha256",
        },
        f"fixture {fixture_id}.expected",
    )
    if expected["rendered_text"] != rendered_text:
        raise ValidationError(f"fixture {fixture_id} rendered text mismatch")
    if expected["token_ids"] != token_ids:
        raise ValidationError(f"fixture {fixture_id} token IDs mismatch")

    rendered_bytes = rendered_text.encode("utf-8")
    encoded_ids = token_ids_bytes(token_ids)
    recomputed = {
        "prompt_tokens": len(token_ids),
        "rendered_text": rendered_text,
        "rendered_text_utf8_bytes": len(rendered_bytes),
        "rendered_text_sha256": sha256_bytes(rendered_bytes),
        "token_ids": token_ids,
        "token_ids_u32le_bytes": len(encoded_ids),
        "token_ids_u32le_sha256": sha256_bytes(encoded_ids),
    }
    if expected != recomputed:
        raise ValidationError(f"fixture {fixture_id} derived hashes or lengths mismatch")


def validate_fixture_set(
    fixture_dir: Path,
    model_dir: Path,
    *,
    tokenizer: Any | None = None,
    transformers_version: str | None = None,
) -> dict[str, Any]:
    root = absolute_path(fixture_dir)
    model = absolute_path(model_dir)
    if root.is_symlink() or not root.is_dir():
        raise ValidationError(f"fixture directory is absent or is a symlink: {root}")

    manifest_path = root / "manifest.json"
    manifest = load_json(manifest_path)
    if tokenizer is None:
        tokenizer, loaded_version = load_local_tokenizer(model)
        transformers_version = loaded_version
    elif not isinstance(transformers_version, str) or not transformers_version:
        raise ValidationError("an injected tokenizer requires transformers_version")

    identity_before = snapshot_model_identity(model)
    template_before = chat_template_identity(tokenizer)
    validate_manifest_identity(
        manifest,
        identity_before,
        tokenizer,
        transformers_version,
    )

    _, empty_ids = render_messages(tokenizer, [{"role": "user", "content": ""}])
    if len(empty_ids) != BASE_PROMPT_TOKENS:
        raise ValidationError(
            "single-user chat template base changed: "
            f"{len(empty_ids)} != {BASE_PROMPT_TOKENS}"
        )

    definitions = expected_fixture_definitions()
    records = manifest["fixture_files"]
    if not isinstance(records, list) or len(records) != len(definitions):
        raise ValidationError("manifest fixture_files count mismatch")
    record_by_id: dict[str, dict[str, Any]] = {}
    for index, raw_record in enumerate(records):
        record = require_exact_keys(
            raw_record,
            {"file", "bytes", "sha256", "fixture_id", "kind", "prompt_tokens"},
            f"manifest.fixture_files[{index}]",
        )
        fixture_id = record["fixture_id"]
        if not isinstance(fixture_id, str) or fixture_id in record_by_id:
            raise ValidationError("manifest fixture IDs are invalid or duplicated")
        record_by_id[fixture_id] = record

    expected_files = {"manifest.json"}
    for definition in definitions:
        fixture_id = definition["fixture_id"]
        relative = f"fixtures/{fixture_id}.json"
        expected_files.add(relative)
        record = record_by_id.get(fixture_id)
        if record is None:
            raise ValidationError(f"manifest has no record for {fixture_id}")
        if record["file"] != relative or record["kind"] != definition["kind"]:
            raise ValidationError(f"manifest record identity mismatch for {fixture_id}")

        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ValidationError(f"fixture file is absent or is a symlink: {relative}")
        actual_record = file_record(path, relative)
        if record["bytes"] != actual_record["bytes"]:
            raise ValidationError(f"fixture file byte count mismatch: {relative}")
        if record["sha256"] != actual_record["sha256"]:
            raise ValidationError(f"fixture file SHA-256 mismatch: {relative}")

        rendered_text, token_ids = render_messages(tokenizer, definition["messages"])
        if not 1 <= len(token_ids) <= CONTEXT_LENGTH:
            raise ValidationError(f"fixture {fixture_id} prompt length is out of range")
        if definition["kind"] == "exact_length":
            target = definition["construction"]["target_prompt_tokens"]
            repeat = definition["construction"]["repeat"]
            if len(token_ids) != BASE_PROMPT_TOKENS + repeat or len(token_ids) != target:
                raise ValidationError(
                    f"fixture {fixture_id} independently rendered length is not {target}"
                )
        if record["prompt_tokens"] != len(token_ids):
            raise ValidationError(f"manifest prompt length mismatch for {fixture_id}")
        validate_fixture_payload(
            load_json(path),
            definition,
            rendered_text,
            token_ids,
        )

    if set(record_by_id) != {item["fixture_id"] for item in definitions}:
        raise ValidationError("manifest contains an unexpected fixture record")

    actual_files = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValidationError(f"fixture tree contains a symlink: {path}")
        if path.is_file():
            actual_files.add(path.relative_to(root).as_posix())
    if actual_files != expected_files:
        raise ValidationError(
            "fixture tree member set mismatch: "
            f"missing={sorted(expected_files - actual_files)} "
            f"extra={sorted(actual_files - expected_files)}"
        )

    identity_after = snapshot_model_identity(model)
    template_after = chat_template_identity(tokenizer)
    if identity_after != identity_before or template_after != template_before:
        raise ValidationError("tokenizer/model identity changed during validation")

    return {
        "fixture_dir": str(root),
        "fixtures": len(definitions),
        "exact_prompt_lengths": list(EXACT_PROMPT_LENGTHS),
        "manifest_sha256": sha256_file(manifest_path),
    }


def main() -> int:
    args = parse_args()
    try:
        result = validate_fixture_set(args.fixture_dir, args.model_dir)
    except (ValidationError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    lengths = ",".join(str(value) for value in result["exact_prompt_lengths"])
    print(
        "passed=true independent_recompute=true "
        f"fixtures={result['fixtures']} exact_prompt_lengths={lengths} "
        f"manifest_sha256={result['manifest_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
