from __future__ import annotations

import json
from pathlib import Path

import pytest

from ullm_openai_gateway.schemas import MODEL_ID, normalize_chat_request
from ullm_openai_gateway.tokenizer import (
    FrozenQwen3Tokenizer,
    StableIncrementalDecoder,
    TokenizerError,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_ROOT = REPO_ROOT / "tests/fixtures/sq8-serving-v0.1/chat-template/fixtures"
MODEL_DIR = Path("/home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3-14B-FP8")
QWEN35_MODEL_DIR = Path("/home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3.5-9B")


@pytest.fixture(scope="module")
def tokenizer() -> FrozenQwen3Tokenizer:
    if not MODEL_DIR.is_dir():
        pytest.skip("frozen local tokenizer is unavailable")
    loaded = FrozenQwen3Tokenizer.load(MODEL_DIR)
    assert __import__("os").environ["HF_HUB_OFFLINE"] == "1"
    assert __import__("os").environ["TRANSFORMERS_OFFLINE"] == "1"
    return loaded


def test_qwen35_frozen_tokenizer_profile_loads() -> None:
    if not QWEN35_MODEL_DIR.is_dir():
        pytest.skip("frozen local Qwen3.5 tokenizer is unavailable")
    loaded = FrozenQwen3Tokenizer.load(QWEN35_MODEL_DIR, "qwen35-9b")
    assert loaded._tokenizer.__class__.__name__ == "Qwen2Tokenizer"


def test_all_frozen_chat_template_fixtures_match_exactly(
    tokenizer: FrozenQwen3Tokenizer,
) -> None:
    fixtures = sorted(FIXTURE_ROOT.glob("*.json"))
    assert fixtures
    for path in fixtures:
        fixture = json.loads(path.read_text(encoding="utf-8"))
        normalized = normalize_chat_request(
            {"model": MODEL_ID, "messages": fixture["messages"]}
        )
        actual = tokenizer.render(normalized.messages)
        assert actual.rendered_text == fixture["expected"]["rendered_text"], path.name
        assert list(actual.token_ids) == fixture["expected"]["token_ids"], path.name
        assert len(actual.token_ids) == fixture["expected"]["prompt_tokens"], path.name


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("日本語の応答です。", "日本語の応答です。"),
        ("emoji: 😀🚀", "emoji: 😀🚀"),
        ("combining: e\u0301", "combining: é"),
        ("```python\nprint('hello')\n```", "```python\nprint('hello')\n```"),
    ],
)
def test_final_decode_is_stable_for_multilingual_text(
    tokenizer: FrozenQwen3Tokenizer, text: str, expected: str
) -> None:
    raw = tokenizer._tokenizer(text, add_special_tokens=False).input_ids
    assert tokenizer.decode(raw) == expected
    assert tokenizer.decode(raw) == tokenizer.decode(raw)


@pytest.mark.parametrize(
    "text",
    [
        "日本語の応答です。",
        "emoji: 😀🚀",
        "combining: e\u0301",
        "```python\nprint('hello')\n```",
    ],
)
def test_incremental_suffixes_equal_final_decode(
    tokenizer: FrozenQwen3Tokenizer, text: str
) -> None:
    token_ids = tokenizer._tokenizer(text, add_special_tokens=False).input_ids
    decoder = StableIncrementalDecoder(tokenizer)
    chunks = [decoder.push(token_id) for token_id in token_ids]
    chunks.append(decoder.finish())
    assert "".join(chunks) == tokenizer.decode(token_ids)
    assert all("\ufffd" not in chunk for chunk in chunks[:-1])


def test_incremental_decoder_holds_replacement_until_sequence_is_stable() -> None:
    class ByteFallbackTokenizer:
        def decode(self, token_ids: list[int]) -> str:
            return {(1,): "a\ufffd", (1, 2): "aあ"}[tuple(token_ids)]

    decoder = StableIncrementalDecoder(ByteFallbackTokenizer())  # type: ignore[arg-type]
    assert decoder.push(1) == "a"
    assert decoder.push(2) == "あ"
    assert decoder.finish() == ""


def test_incremental_decoder_rejects_changed_emitted_prefix() -> None:
    class UnstableTokenizer:
        def decode(self, token_ids: list[int]) -> str:
            return {(1,): "stable", (1, 2): "changed"}[tuple(token_ids)]

    decoder = StableIncrementalDecoder(UnstableTokenizer())  # type: ignore[arg-type]
    assert decoder.push(1) == "stable"
    with pytest.raises(TokenizerError, match="changed an emitted prefix"):
        decoder.push(2)


def test_incremental_decoder_rejects_invalid_or_post_finish_push() -> None:
    class EmptyTokenizer:
        def decode(self, _: list[int]) -> str:
            return ""

    invalid = StableIncrementalDecoder(EmptyTokenizer())  # type: ignore[arg-type]
    with pytest.raises(TokenizerError, match="invalid token ID"):
        invalid.push(-1)
    finished = StableIncrementalDecoder(EmptyTokenizer())  # type: ignore[arg-type]
    assert finished.finish() == ""
    with pytest.raises(TokenizerError, match="already finished"):
        finished.push(1)
