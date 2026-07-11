"""Frozen local Qwen3 tokenizer and stable final decoding."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .schemas import NormalizedMessage


EXPECTED_TRANSFORMERS_VERSION = "5.12.1"
EXPECTED_TOKENIZER_CLASS = "Qwen2Tokenizer"
EXPECTED_CHAT_TEMPLATE_SHA256 = (
    "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
)
EXPECTED_FILES = {
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "generation_config.json": "231c22c0b89ffbbb785d0e68b2f3f922244f263487af79f6542fc82dbee37dbf",
}


class TokenizerError(RuntimeError):
    """Raised when frozen tokenizer identity or behavior differs."""


@dataclass(frozen=True, slots=True)
class TokenizedPrompt:
    rendered_text: str
    token_ids: tuple[int, ...]


class FrozenQwen3Tokenizer:
    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    @classmethod
    def load(cls, directory: Path) -> "FrozenQwen3Tokenizer":
        _validate_files(directory)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        try:
            import transformers
            from transformers import AutoTokenizer
        except ImportError as error:
            raise TokenizerError(
                "the pinned Transformers package is unavailable"
            ) from error
        if transformers.__version__ != EXPECTED_TRANSFORMERS_VERSION:
            raise TokenizerError("Transformers version differs from the frozen version")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                directory,
                local_files_only=True,
                trust_remote_code=False,
            )
        except Exception as error:
            raise TokenizerError("failed to load the frozen local tokenizer") from error
        if tokenizer.__class__.__name__ != EXPECTED_TOKENIZER_CLASS:
            raise TokenizerError("tokenizer class differs from the frozen class")
        template = getattr(tokenizer, "chat_template", None)
        if (
            not isinstance(template, str)
            or _sha256_bytes(template.encode("utf-8")) != EXPECTED_CHAT_TEMPLATE_SHA256
        ):
            raise TokenizerError("chat template differs from the frozen template")
        return cls(tokenizer)

    def render(self, messages: Iterable[NormalizedMessage]) -> TokenizedPrompt:
        normalized = [message.as_template_value() for message in messages]
        options = {"add_generation_prompt": True, "enable_thinking": False}
        try:
            rendered = self._tokenizer.apply_chat_template(
                normalized,
                tokenize=False,
                **options,
            )
            token_ids = self._tokenizer.apply_chat_template(
                normalized,
                tokenize=True,
                **options,
            )
        except Exception as error:
            raise TokenizerError("chat template application failed") from error
        if not isinstance(rendered, str):
            raise TokenizerError("chat template did not produce text")
        ids = _extract_token_ids(token_ids)
        if not ids:
            raise TokenizerError("chat template produced no token IDs")
        return TokenizedPrompt(rendered, tuple(ids))

    def decode(self, token_ids: Iterable[int]) -> str:
        ids = list(token_ids)
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in ids
        ):
            raise TokenizerError("generated token IDs are invalid")
        try:
            value = self._tokenizer.decode(
                ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        except Exception as error:
            raise TokenizerError("final token decoding failed") from error
        if not isinstance(value, str):
            raise TokenizerError("final token decoding did not produce text")
        try:
            value.encode("utf-8", errors="strict")
        except UnicodeEncodeError as error:
            raise TokenizerError(
                "final token decoding produced invalid Unicode"
            ) from error
        return value


class StableIncrementalDecoder:
    """Emit only decoded prefixes that cannot contain incomplete UTF-8."""

    def __init__(self, tokenizer: FrozenQwen3Tokenizer) -> None:
        self._tokenizer = tokenizer
        self._token_ids: list[int] = []
        self._emitted = ""
        self._finished = False

    def push(self, token_id: int) -> str:
        if self._finished:
            raise TokenizerError("incremental decoder is already finished")
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            raise TokenizerError("incremental decoder received an invalid token ID")
        self._token_ids.append(token_id)
        decoded = self._tokenizer.decode(self._token_ids)
        if not decoded.startswith(self._emitted):
            raise TokenizerError("incremental decode changed an emitted prefix")
        replacement = decoded.find("\ufffd", len(self._emitted))
        stable = decoded if replacement < 0 else decoded[:replacement]
        suffix = stable[len(self._emitted) :]
        self._emitted = stable
        return suffix

    def finish(self) -> str:
        if self._finished:
            raise TokenizerError("incremental decoder is already finished")
        self._finished = True
        decoded = self._tokenizer.decode(self._token_ids)
        if not decoded.startswith(self._emitted):
            raise TokenizerError("final decode changed an emitted prefix")
        suffix = decoded[len(self._emitted) :]
        self._emitted = decoded
        return suffix

    @property
    def text(self) -> str:
        return self._emitted


def _validate_files(directory: Path) -> None:
    if not directory.is_dir():
        raise TokenizerError("tokenizer directory is absent")
    for name, expected in EXPECTED_FILES.items():
        path = directory / name
        if not path.is_file() or path.is_symlink():
            raise TokenizerError("a frozen tokenizer file is absent or is a symlink")
        if _sha256_file(path) != expected:
            raise TokenizerError("a frozen tokenizer file hash differs")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _extract_token_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        if len(value) != 1:
            raise TokenizerError("tokenizer returned multiple prompt sequences")
        value = value[0]
    if not isinstance(value, (list, tuple)):
        raise TokenizerError("tokenizer output is not a token-ID sequence")
    result: list[int] = []
    for item in value:
        if (
            isinstance(item, bool)
            or not isinstance(item, int)
            or not 0 <= item <= 0xFFFF_FFFF
        ):
            raise TokenizerError("tokenizer returned an invalid token ID")
        result.append(item)
    return result
