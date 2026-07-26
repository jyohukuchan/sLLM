from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/lightweight_promotion.py"


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PROMOTION = load_module("test_lightweight_promotion_tool", TOOL)


def canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def test_prompt_suite_has_required_coverage() -> None:
    suite = PROMOTION.load_suite(PROMOTION.DEFAULT_PROMPT_SUITE)

    assert 8 <= len(suite) <= 16
    assert {case.category for case in suite} >= {
        "japanese",
        "english",
        "code_generation",
        "long_summary",
        "multi_turn",
    }


def test_strict_json_rejects_duplicate_keys_and_nonfinite_values() -> None:
    with pytest.raises(PROMOTION.PromotionError, match="duplicate key"):
        PROMOTION.strict_object(b'{"same":1,"same":2}', "fixture")
    with pytest.raises(PROMOTION.PromotionError, match="non-finite"):
        PROMOTION.strict_object(b'{"value":NaN}', "fixture")


def test_text_collapse_and_response_abandonment_are_blocking() -> None:
    code_case = PROMOTION.SuiteCase(
        case_id="code",
        category="code_generation",
        messages=({"role": "user", "content": "write code"},),
        max_completion_tokens=32,
        expected_language="en",
        expected_kind="code",
    )
    japanese_case = PROMOTION.SuiteCase(
        case_id="ja",
        category="japanese",
        messages=({"role": "user", "content": "日本語で答える"},),
        max_completion_tokens=32,
        expected_language="ja",
        expected_kind="prose",
    )

    assert "code_structure_not_observed" in PROMOTION.analyze_text(
        "This is an explanation without source syntax.", code_case
    )["blocking"]
    assert "expected_japanese_not_observed" in PROMOTION.analyze_text(
        "This answer contains enough English letters to clearly be unrelated to Japanese output.",
        japanese_case,
    )["blocking"]
    assert "repeated_sentence_loop" in PROMOTION.analyze_text(
        "The same sentence repeats. The same sentence repeats. The same sentence repeats.",
        code_case,
    )["blocking"]


def test_atomic_exchange_preserves_exact_bytes_and_rejects_wrong_expected_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    active = tmp_path / "active.json"
    old = b'{"old":true}\n'
    new = b'{"new":true}\n'
    active.write_bytes(old)
    active.chmod(0o640)
    monkeypatch.setattr(PROMOTION, "_require_active_parent", lambda _path: None)

    assert PROMOTION.atomic_switch(active, old, new) is True
    assert active.read_bytes() == new
    assert PROMOTION.atomic_switch(active, new, old) is True
    assert active.read_bytes() == old
    with pytest.raises(PROMOTION.PromotionError, match="changed concurrently"):
        PROMOTION.atomic_switch(active, b"not-the-active-bytes", new)
    assert active.read_bytes() == old
    assert not list(tmp_path.glob(".active.json.lightweight-stage-*"))


def test_rollback_preflight_requires_candidate_bytes_to_differ_from_saved_rollback(
    tmp_path: Path
) -> None:
    active = tmp_path / "active.json"
    saved_rollback = tmp_path / "rollback-active.json"
    candidate = b'{"candidate":true}\n'
    old = b'{"old":true}\n'
    active.write_bytes(candidate)
    saved_rollback.write_bytes(old)
    candidate_hash = hashlib.sha256(candidate).hexdigest()
    rollback_hash = hashlib.sha256(old).hexdigest()
    transaction = tmp_path / "transaction.json"
    transaction.write_bytes(
        canonical(
            {
                "schema_version": PROMOTION.PROMOTION_TRANSACTION_SCHEMA,
                "run_id": "fixture-run",
                "active_manifest_path": os.fspath(active),
                "candidate_manifest_sha256": candidate_hash,
                "rollback_manifest_path": os.fspath(saved_rollback),
                "rollback_manifest_sha256": rollback_hash,
                "base_url": "http://127.0.0.1:8000",
                "service": "fixture.service",
            }
        )
    )
    transaction_hash = hashlib.sha256(transaction.read_bytes()).hexdigest()
    outcome = tmp_path / "activation-outcome.json"
    outcome.write_bytes(
        canonical(
            {
                "schema_version": PROMOTION.PROMOTION_OUTCOME_SCHEMA,
                "status": "activated",
                "transaction_path": os.fspath(transaction),
                "transaction_sha256": transaction_hash,
                "candidate_manifest_sha256": candidate_hash,
                "rollback_manifest_sha256": rollback_hash,
            }
        )
    )
    args = SimpleNamespace(activation_outcome=outcome, yes=False)

    ready = PROMOTION.rollback(args)
    assert ready["ready"] is True
    assert ready["strict_byte_difference"] is True

    active.write_bytes(old)
    rejected = PROMOTION.rollback(args)
    assert rejected["ready"] is False
    assert rejected["strict_byte_difference"] is False

    tampered = json.loads(outcome.read_text(encoding="utf-8"))
    tampered["transaction_sha256"] = "0" * 64
    outcome.write_bytes(canonical(tampered))
    with pytest.raises(PROMOTION.PromotionError, match="transaction hash"):
        PROMOTION.rollback(args)


def test_semantic_self_test_is_machine_confirmed() -> None:
    with pytest.raises(SystemExit):
        PROMOTION.parse_promotion_args(
            ["--candidate-manifest", "/tmp/candidate.json", "--evidence-dir", "/tmp/evidence", "--semantic-self-test"]
        )
