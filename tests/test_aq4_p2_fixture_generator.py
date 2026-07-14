from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("aq4_fixture_generator", ROOT / "tools/generate-aq4-p2-fixtures.py")
assert SPEC and SPEC.loader
GEN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GEN)


def _case(case_id: str, stage: str, prompt: int, generated: int = 0) -> dict:
    value = {
        "case_id": case_id,
        "fixture_id": case_id,
        "case_sha256": None,
        "stage_id": stage,
        "stage_order": 1,
        "scope": "full_model",
        "phase": "cold_prefill",
        "mode": "cold_batched",
        "baseline_mode": "cold_batched",
        "prompt_tokens": prompt,
        "cached_prefix_tokens": 0,
        "context_tokens": prompt,
        "decode_start_tokens": 0,
        "prefill_requested_m": 8,
        "resolved_m": 8,
        "request_count": 1,
        "decode_request_count": 0,
        "generated_tokens": generated,
        "device": {"device_id": "cpu-reference", "backend": "cpu", "name": "cpu", "architecture": "x86", "runtime_device_index": -1},
        "control_id": "aq4_0_target",
        "control": {"control_id": "aq4_0_target"},
        "sampling": {"mode": "greedy"},
        "format_id": "AQ4_0",
        "implementation_id": "qwen35_aq4_rdna4_v1",
        "path_oracle_case_id": None,
        "path_oracle_result_sha256": None,
    }
    value["case_sha256"] = GEN.case_hash(value)
    return value


def _served(root: Path) -> dict:
    tokenizer = root / "tokenizer"
    tokenizer.mkdir()
    payload = b"synthetic tokenizer fixture"
    (tokenizer / "tokenizer.json").write_bytes(payload)
    return {
        "schema_version": "ullm.served_model.v2",
        "public": {"context_length": 64},
        "generation": {"max_completion_tokens": 16, "vocab_size": 128, "eos_token_ids": [127]},
        "tokenizer": {"root": str(tokenizer), "files": {"tokenizer.json": hashlib.sha256(payload).hexdigest()}},
        "reasoning": {"start_token_ids": [126], "end_token_ids": [125], "forced_end_token_ids": [124]},
    }


def test_generation_is_hash_bound_and_hides_token_ids(tmp_path: Path) -> None:
    served_value = _served(tmp_path)
    served = GEN.validate_served_model(served_value, tmp_path)
    cases = [_case("smoke-a", "smoke", 9), _case("representative-a", "representative", 3, 2)]
    expanded = {"schema_version": "ullm.aq4_production_p2_expanded.v2", "case_count": 2, "stage_case_count": {"smoke": 1, "representative": 1}, "cases": cases}
    output = tmp_path / "fixtures"
    index = GEN.generate(expanded, "a" * 64, served, "b" * 64, output, "all")
    assert index["case_count"] == 2
    assert index["contract"]["reserved_token_ids_sha256"] == served["reserved_token_ids_sha256"]
    text = json.dumps(index)
    assert '"prompt_token_ids":' not in text
    fixture = json.loads((output / "cases" / "smoke-a.json").read_text())
    assert fixture["cases"][0]["step_count"] == 0
    assert not set(fixture["cases"][0]["prompt_token_ids"]) & {124, 125, 126, 127}
    smoke_entry = next(item for item in index["cases"] if item["case_id"] == "smoke-a")
    assert smoke_entry["prompt_token_ids_sha256"] == hashlib.sha256(GEN.canonical(fixture["cases"][0]["prompt_token_ids"])).hexdigest()


def test_subset_is_exact_and_symlink_tokenizer_is_rejected(tmp_path: Path) -> None:
    served_value = _served(tmp_path)
    served = GEN.validate_served_model(served_value, tmp_path)
    case = _case("smoke-a", "smoke", 2)
    expanded = {"schema_version": "ullm.aq4_production_p2_expanded.v2", "case_count": 1, "stage_case_count": {"smoke": 2}, "cases": [case]}
    with pytest.raises(GEN.FixtureError, match="subset case count"):
        GEN.generate(expanded, "a" * 64, served, "b" * 64, tmp_path / "fixtures", "smoke")
    link_root = tmp_path / "link-root"
    link_root.symlink_to(tmp_path / "tokenizer", target_is_directory=True)
    bad = dict(served_value)
    bad["tokenizer"] = {"root": str(link_root), "files": served_value["tokenizer"]["files"]}
    with pytest.raises(GEN.FixtureError, match="symlink"):
        GEN.validate_served_model(bad, tmp_path)


def test_fixture_writer_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "index.json"
    GEN.atomic_write(path, {"ok": True})
    with pytest.raises(GEN.FixtureError, match="overwrite"):
        GEN.atomic_write(path, {"ok": False})
