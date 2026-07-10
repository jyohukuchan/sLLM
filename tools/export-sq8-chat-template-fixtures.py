#!/usr/bin/env python3
"""Export frozen Qwen3 chat-template fixtures for the SQ8 serving contract."""

from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import hashlib
import json
import operator
import os
import struct
import sys
import tempfile
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
AT_FDCWD = -100
RENAME_NOREPLACE = 1


class ExportError(RuntimeError):
    """Raised when a fixture set cannot be exported without weakening its contract."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    path.write_text(payload, encoding="ascii")


def ensure_output_available(output_dir: Path) -> None:
    if os.path.lexists(output_dir):
        raise ExportError(f"refusing to overwrite existing output: {output_dir}")


def rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a sibling directory without replacing a raced path."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise ExportError("renameat2 is required for atomic no-clobber publication")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        AT_FDCWD,
        os.fsencode(source),
        AT_FDCWD,
        os.fsencode(destination),
        RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(f"refusing to overwrite raced output: {destination}")
    raise OSError(error_number, os.strerror(error_number), str(destination))


def read_revision(model_dir: Path, filename: str) -> str:
    metadata = model_dir / ".cache" / "huggingface" / "download" / f"{filename}.metadata"
    try:
        lines = metadata.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ExportError(f"missing revision metadata for {filename}: {error}") from error
    if not lines or lines[0] != EXPECTED_REVISION:
        actual = lines[0] if lines else None
        raise ExportError(
            f"revision mismatch for {filename}: {actual!r} != {EXPECTED_REVISION!r}"
        )
    return lines[0]


def snapshot_model_identity(model_dir: Path) -> dict[str, Any]:
    if not model_dir.is_dir():
        raise ExportError(f"model directory does not exist: {model_dir}")

    config_path = model_dir / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExportError(f"failed to read model config: {error}") from error
    if not isinstance(config, dict) or config.get("model_type") != "qwen3":
        raise ExportError("model config is not the fixed Qwen3 model type")

    tokenizer_names = list(TOKENIZER_REQUIRED_FILES)
    for name in TOKENIZER_OPTIONAL_FILES:
        if (model_dir / name).is_file():
            tokenizer_names.append(name)

    tokenizer_files = []
    for name in tokenizer_names:
        path = model_dir / name
        if not path.is_file():
            raise ExportError(f"missing tokenizer file: {path}")
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
        raise ExportError(f"transformers is unavailable: {error}") from error
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as error:
        raise ExportError(f"failed to load the local tokenizer: {error}") from error
    return tokenizer, str(transformers.__version__)


def chat_template_identity(tokenizer: Any) -> dict[str, Any]:
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template:
        raise ExportError("tokenizer.chat_template must be a non-empty string")
    encoded = template.encode("utf-8")
    return {
        "utf8_bytes": len(encoded),
        "sha256": sha256_bytes(encoded),
    }


def extract_input_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise ExportError("tokenized chat template has no input_ids")
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ExportError("tokenized chat template input_ids is not a sequence")
    if value and isinstance(value[0], Sequence):
        if len(value) != 1:
            raise ExportError("tokenized chat template returned more than one sequence")
        value = value[0]

    result = []
    for index, item in enumerate(value):
        if isinstance(item, bool):
            raise ExportError(f"token ID {index} is boolean")
        try:
            token_id = operator.index(item)
        except TypeError as error:
            raise ExportError(f"token ID {index} is not an integer") from error
        if token_id < 0 or token_id > 0xFFFF_FFFF:
            raise ExportError(f"token ID {index} is outside unsigned 32-bit range")
        result.append(int(token_id))
    if not result:
        raise ExportError("chat template produced no token IDs")
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
        raise ExportError("non-tokenized chat template output is not a string")
    encoded = tokenizer.apply_chat_template(
        copy.deepcopy(messages),
        tokenize=True,
        **keyword_args,
    )
    return rendered, extract_input_ids(encoded)


def fixture_definitions() -> list[dict[str, Any]]:
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


def fixture_payload(
    definition: dict[str, Any], rendered_text: str, token_ids: list[int]
) -> dict[str, Any]:
    rendered_bytes = rendered_text.encode("utf-8")
    encoded_ids = token_ids_bytes(token_ids)
    return {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "fixture_id": definition["fixture_id"],
        "kind": definition["kind"],
        "messages": definition["messages"],
        "construction": definition["construction"],
        "template_options": {
            "add_generation_prompt": True,
            "enable_thinking": False,
        },
        "expected": {
            "prompt_tokens": len(token_ids),
            "rendered_text": rendered_text,
            "rendered_text_utf8_bytes": len(rendered_bytes),
            "rendered_text_sha256": sha256_bytes(rendered_bytes),
            "token_ids": token_ids,
            "token_ids_u32le_bytes": len(encoded_ids),
            "token_ids_u32le_sha256": sha256_bytes(encoded_ids),
        },
    }


def build_staged_fixture_set(
    root: Path,
    *,
    tokenizer: Any,
    transformers_version: str,
    identity: dict[str, Any],
) -> dict[str, Any]:
    empty_text, empty_ids = render_messages(
        tokenizer, [{"role": "user", "content": ""}]
    )
    del empty_text
    if len(empty_ids) != BASE_PROMPT_TOKENS:
        raise ExportError(
            "single-user chat template base changed: "
            f"{len(empty_ids)} != {BASE_PROMPT_TOKENS}"
        )

    records = []
    for definition in fixture_definitions():
        rendered_text, token_ids = render_messages(tokenizer, definition["messages"])
        if not 1 <= len(token_ids) <= CONTEXT_LENGTH:
            raise ExportError(
                f"fixture {definition['fixture_id']} prompt length is out of range"
            )
        if definition["kind"] == "exact_length":
            target = definition["construction"]["target_prompt_tokens"]
            repeat = definition["construction"]["repeat"]
            if len(token_ids) != BASE_PROMPT_TOKENS + repeat or len(token_ids) != target:
                raise ExportError(
                    f"fixture {definition['fixture_id']} has {len(token_ids)} tokens; "
                    f"expected exact length {target}"
                )

        relative = f"fixtures/{definition['fixture_id']}.json"
        path = root / relative
        write_json(path, fixture_payload(definition, rendered_text, token_ids))
        record = file_record(path, relative)
        record.update(
            {
                "fixture_id": definition["fixture_id"],
                "kind": definition["kind"],
                "prompt_tokens": len(token_ids),
            }
        )
        records.append(record)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "fixture_set_id": FIXTURE_SET_ID,
        "model": identity["model"],
        "tokenizer": {
            "revision": identity["tokenizer_revision"],
            "class": type(tokenizer).__name__,
            "transformers_version": transformers_version,
            "loader": {
                "local_files_only": True,
                "trust_remote_code": False,
            },
            "files": identity["tokenizer_files"],
            "revision_evidence": identity["revision_evidence"],
            "chat_template": chat_template_identity(tokenizer),
        },
        "template_options": {
            "add_generation_prompt": True,
            "enable_thinking": False,
        },
        "exact_length_contract": {
            "base_prompt_tokens": BASE_PROMPT_TOKENS,
            "message_shape": "single user",
            "content_rule": "' x' * repeat",
            "target_prompt_tokens": list(EXACT_PROMPT_LENGTHS),
        },
        "fixture_files": records,
    }
    write_json(root / "manifest.json", manifest)
    return manifest


def export_fixture_set(
    output_dir: Path,
    model_dir: Path,
    *,
    tokenizer: Any | None = None,
    transformers_version: str | None = None,
) -> dict[str, Any]:
    output = absolute_path(output_dir)
    model = absolute_path(model_dir)
    ensure_output_available(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if tokenizer is None:
        tokenizer, loaded_version = load_local_tokenizer(model)
        transformers_version = loaded_version
    elif not isinstance(transformers_version, str) or not transformers_version:
        raise ExportError("an injected tokenizer requires transformers_version")

    identity_before = snapshot_model_identity(model)
    template_before = chat_template_identity(tokenizer)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temp:
        staged = Path(temp) / output.name
        staged.mkdir()
        manifest = build_staged_fixture_set(
            staged,
            tokenizer=tokenizer,
            transformers_version=transformers_version,
            identity=identity_before,
        )
        identity_after = snapshot_model_identity(model)
        template_after = chat_template_identity(tokenizer)
        if identity_after != identity_before or template_after != template_before:
            raise ExportError("tokenizer/model identity changed during fixture export")
        rename_noreplace(staged, output)

    return {
        "output": str(output),
        "fixtures": len(manifest["fixture_files"]),
        "manifest_sha256": sha256_file(output / "manifest.json"),
    }


def main() -> int:
    args = parse_args()
    try:
        result = export_fixture_set(args.output_dir, args.model_dir)
    except (ExportError, FileExistsError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        "exported=true "
        f"output={result['output']} fixtures={result['fixtures']} "
        f"manifest_sha256={result['manifest_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
