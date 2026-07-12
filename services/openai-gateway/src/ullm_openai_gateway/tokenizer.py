"""Frozen local Qwen3 tokenizer and stable final decoding."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .schemas import NormalizedMessage
from .served_model import TokenizerContract


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

QWEN35_EXPECTED_CHAT_TEMPLATE_SHA256 = (
    "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"
)
QWEN35_EXPECTED_FILES = {
    "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    "tokenizer_config.json": "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8",
    "vocab.json": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
    "merges.txt": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
}

TOKENIZER_PROFILES = {
    "qwen3-14b": (
        EXPECTED_TOKENIZER_CLASS,
        EXPECTED_CHAT_TEMPLATE_SHA256,
        EXPECTED_FILES,
    ),
    "qwen35-9b": (
        EXPECTED_TOKENIZER_CLASS,
        QWEN35_EXPECTED_CHAT_TEMPLATE_SHA256,
        QWEN35_EXPECTED_FILES,
    ),
}


class TokenizerError(RuntimeError):
    """Raised when frozen tokenizer identity or behavior differs."""


@dataclass(frozen=True, slots=True)
class TokenizedPrompt:
    rendered_text: str
    token_ids: tuple[int, ...]


class FrozenQwen3Tokenizer:
    def __init__(
        self,
        tokenizer: Any,
        *,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
    ) -> None:
        self._tokenizer = tokenizer
        self._template_options = {
            "add_generation_prompt": add_generation_prompt,
            "enable_thinking": enable_thinking,
        }

    @classmethod
    def load(
        cls, directory: Path, profile: str = "qwen3-14b"
    ) -> "FrozenQwen3Tokenizer":
        try:
            expected_class, expected_template, expected_files = TOKENIZER_PROFILES[
                profile
            ]
        except KeyError as error:
            raise TokenizerError("the tokenizer profile is unsupported") from error
        return cls._load_identity(
            directory,
            transformers_version=EXPECTED_TRANSFORMERS_VERSION,
            expected_class=expected_class,
            expected_template=expected_template,
            expected_files=expected_files,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    @classmethod
    def load_contract(cls, contract: TokenizerContract) -> "FrozenQwen3Tokenizer":
        return cls._load_identity(
            contract.root,
            transformers_version=contract.transformers_version,
            expected_class=contract.class_name,
            expected_template=contract.chat_template_sha256,
            expected_files={item.path: item.sha256 for item in contract.files},
            add_generation_prompt=contract.add_generation_prompt,
            enable_thinking=contract.enable_thinking,
        )

    @classmethod
    def _load_identity(
        cls,
        directory: Path,
        *,
        transformers_version: str,
        expected_class: str,
        expected_template: str,
        expected_files: Mapping[str, str],
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> "FrozenQwen3Tokenizer":
        _validate_files(directory, expected_files)
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
        if transformers.__version__ != transformers_version:
            raise TokenizerError("Transformers version differs from the frozen version")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                directory,
                local_files_only=True,
                trust_remote_code=False,
            )
        except Exception as error:
            raise TokenizerError("failed to load the frozen local tokenizer") from error
        if tokenizer.__class__.__name__ != expected_class:
            raise TokenizerError("tokenizer class differs from the frozen class")
        template = getattr(tokenizer, "chat_template", None)
        if (
            not isinstance(template, str)
            or _sha256_bytes(template.encode("utf-8")) != expected_template
        ):
            raise TokenizerError("chat template differs from the frozen template")
        return cls(
            tokenizer,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )

    def render(self, messages: Iterable[NormalizedMessage]) -> TokenizedPrompt:
        normalized = [message.as_template_value() for message in messages]
        try:
            rendered = self._tokenizer.apply_chat_template(
                normalized,
                tokenize=False,
                **self._template_options,
            )
            token_ids = self._tokenizer.apply_chat_template(
                normalized,
                tokenize=True,
                **self._template_options,
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


def _validate_files(directory: Path, expected_files: Mapping[str, str]) -> None:
    if not directory.is_dir():
        raise TokenizerError("tokenizer directory is absent")
    for name, expected in expected_files.items():
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
