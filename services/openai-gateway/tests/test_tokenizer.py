from __future__ import annotations

import json
from pathlib import Path

import pytest

from ullm_openai_gateway.schemas import MODEL_ID, normalize_chat_request
from ullm_openai_gateway.tokenizer import FrozenQwen3Tokenizer


REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_ROOT = REPO_ROOT / "tests/fixtures/sq8-serving-v0.1/chat-template/fixtures"
MODEL_DIR = Path("/home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3-14B-FP8")


@pytest.fixture(scope="module")
def tokenizer() -> FrozenQwen3Tokenizer:
    if not MODEL_DIR.is_dir():
        pytest.skip("frozen local tokenizer is unavailable")
    loaded = FrozenQwen3Tokenizer.load(MODEL_DIR)
    assert __import__("os").environ["HF_HUB_OFFLINE"] == "1"
    assert __import__("os").environ["TRANSFORMERS_OFFLINE"] == "1"
    return loaded


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
