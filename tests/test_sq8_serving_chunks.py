import copy
import hashlib
import importlib.util
import json
import shutil
import struct
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = ROOT / "tests/fixtures/sq8-serving-v0.1/oracles/vllm-source-v0.1"
TOOL_PATH = ROOT / "tools/validate-sq8-serving-chunks.py"
DEFAULT_CHUNK_MODE = "m8-chunk8"
CHUNK_MODES = {
    DEFAULT_CHUNK_MODE: {
        "schema_version": "ullm.sq8.serving_chunks.v3",
        "prefill_chunk_tokens": 8,
        "prefill_implementation": "sq8.fixed-m8-cached-prefix.v1",
    },
    "m32-chunk32": {
        "schema_version": "ullm.sq8.serving_chunks.v4",
        "prefill_chunk_tokens": 32,
        "prefill_implementation": "sq8.fixed-m32-cached-prefix.v1",
    },
    "m128-chunk128": {
        "schema_version": "ullm.sq8.serving_chunks.v4",
        "prefill_chunk_tokens": 128,
        "prefill_implementation": "sq8.fixed-m128-cached-prefix.v1",
    },
}


def load_tool():
    spec = importlib.util.spec_from_file_location("validate_sq8_serving_chunks", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_top1(prompt_tokens: int = 8) -> tuple[int, float]:
    prompt = FIXTURE_ROOT / f"prompts/raw-p{prompt_tokens:04d}"
    token_id = struct.unpack("<I", (prompt / "greedy-g1.u32le").read_bytes())[0]
    logits = struct.iter_unpack("<f", (prompt / "prefill-logits.f32le").read_bytes())
    top_token, top_logit = max(
        enumerate(value[0] for value in logits), key=lambda item: (item[1], -item[0])
    )
    assert top_token == token_id
    return token_id, top_logit


def unit_trace(widths: list[int]) -> list[dict]:
    result = []
    position = 0
    for index, width in enumerate(widths):
        end = position + width
        result.append(
            {
                "start_position": position,
                "width": width,
                "end_position": end,
                "final_prompt_unit": index + 1 == len(widths),
                "cache_lengths": [end] * 40,
                "cache_lengths_all_expected": True,
                "last_cache_position": end - 1,
                "last_logical_block": (end - 1) // 16,
            }
        )
        position = end
    return result


def result_document(
    mode: str,
    capture_prefix: str,
    token_id: int,
    top1_logit: float,
    *,
    prompt_tokens: int = 8,
    chunk_mode: str = DEFAULT_CHUNK_MODE,
) -> dict:
    chunk_config = CHUNK_MODES[chunk_mode]
    chunk_tokens = chunk_config["prefill_chunk_tokens"]
    widths = (
        [1] * prompt_tokens
        if mode == "all-m1"
        else [chunk_tokens] * (prompt_tokens // chunk_tokens)
        + [1] * (prompt_tokens % chunk_tokens)
    )
    schema = (
        "ullm.sq8.serving_smoke.v2"
        if mode == "all-m1"
        else chunk_config["schema_version"]
    )
    implementation = (
        "sq8.sequential-m1.v1"
        if mode == "all-m1"
        else chunk_config["prefill_implementation"]
    )
    return {
        "schema_version": schema,
        "passed": False,
        "prefill_mode": mode,
        "prefill_chunk_tokens": 8 if mode == "all-m1" else chunk_tokens,
        "prefill_implementation": implementation,
        "artifact_content_sha256": (
            "2243acf1df627ff6ec13840c8ffcf35c77e89205eb36cef7561b85c9c98b9147"
        ),
        "package_manifest_sha256": (
            "c2133dfe392f3d5608bde17ed764ae8347c3096c500a58aa235adbeb63d1a0eb"
        ),
        "device": {
            "device_id": 0,
            "backend": "hip",
            "name": "AMD Radeon Graphics",
            "gcn_arch_name": "gfx1201",
            "compute_major": 12,
            "compute_minor": 0,
            "total_global_mem": 34_208_743_424,
        },
        "kv_cache_bytes": 1_342_177_280,
        "cache_blocks": 256,
        "context_tokens": 4096,
        "post_reset_status": "ready",
        "post_reset_active": 0,
        "post_reset_waiting": 0,
        "post_reset_allocated_blocks": 0,
        "post_reset_cache_lengths": [0] * 40,
        "post_reset_cache_lengths_all_zero": True,
        "requests": [
            {
                "request_id": f"synthetic-{mode}",
                "prompt_token_ids": list(range(1, prompt_tokens + 1)),
                "max_new_tokens": 1,
                "generated_token_ids": [token_id],
                "prompt_progress_events": len(widths) - 1,
                "execution_units": len(widths),
                "processed_prompt_tokens": prompt_tokens,
                "execution_calls": len(widths),
                "prefill_execution_units": unit_trace(widths),
                "terminal_expected_cache_len": prompt_tokens,
                "terminal_cache_lengths": [prompt_tokens] * 40,
                "terminal_cache_lengths_all_expected": True,
                "terminal_last_cache_position": prompt_tokens - 1,
                "terminal_last_logical_block": (prompt_tokens - 1) // 16,
                "terminal_scheduler_active": 1,
                "terminal_scheduler_waiting": 0,
                "terminal_allocated_blocks": 256,
                "terminal_reason": "length",
                "release_outcome": "length",
                "oracle_capture": {
                    "position": prompt_tokens - 1,
                    "top1_token_id": token_id,
                    "top1_logit": top1_logit,
                    "final_hidden_file": f"{capture_prefix}-hidden.f32le",
                    "final_hidden_f32_le_sha256": "",
                    "logits_file": f"{capture_prefix}-logits.f32le",
                    "logits_f32_le_sha256": "",
                },
            }
        ],
    }


def write_fixture(
    tmp_path: Path,
    *,
    chunk_mode: str = DEFAULT_CHUNK_MODE,
    prompt_tokens: int = 8,
) -> tuple[Path, Path]:
    token_id, top1_logit = source_top1(prompt_tokens)
    source = FIXTURE_ROOT / f"prompts/raw-p{prompt_tokens:04d}"
    documents = []
    for mode, prefix in [(chunk_mode, "chunk"), ("all-m1", "m1")]:
        hidden = tmp_path / f"{prefix}-hidden.f32le"
        logits = tmp_path / f"{prefix}-logits.f32le"
        shutil.copyfile(source / "final-hidden.f32le", hidden)
        shutil.copyfile(source / "prefill-logits.f32le", logits)
        document = result_document(
            mode,
            prefix,
            token_id,
            top1_logit,
            prompt_tokens=prompt_tokens,
            chunk_mode=chunk_mode,
        )
        capture = document["requests"][0]["oracle_capture"]
        capture["final_hidden_f32_le_sha256"] = sha256_file(hidden)
        capture["logits_f32_le_sha256"] = sha256_file(logits)
        result = tmp_path / f"{prefix}.json"
        result.write_text(json.dumps(document), encoding="utf-8")
        documents.append(result)
    return documents[0], documents[1]


def validate(
    tool,
    chunk: Path,
    m1: Path,
    *,
    chunk_mode: str = DEFAULT_CHUNK_MODE,
    prompt_tokens: int = 8,
):
    return tool.validate_results(
        chunk,
        m1,
        FIXTURE_ROOT,
        (prompt_tokens,),
        (prompt_tokens,),
        chunk_mode=chunk_mode,
    )


def test_chunk_validator_recomputes_all_gates(tmp_path):
    tool = load_tool()
    chunk, m1 = write_fixture(tmp_path)
    result = validate(tool, chunk, m1)
    assert result["passed"] is True
    assert result["prompts"][0]["chunk_vs_all_m1"]["logits"]["top_1_exact"] is True
    assert result["prompts"][0]["chunk_vs_source"]["logits"]["top_10_overlap"] == 10
    assert result["evidence"][0]["schema_version"] == "ullm.sq8.serving_chunks.v3"
    assert result["evidence"][0]["prefill_mode"] == "m8-chunk8"


@pytest.mark.parametrize(
    ("chunk_mode", "prompt_tokens", "chunk_tokens", "implementation"),
    [
        ("m32-chunk32", 32, 32, "sq8.fixed-m32-cached-prefix.v1"),
        ("m128-chunk128", 128, 128, "sq8.fixed-m128-cached-prefix.v1"),
    ],
)
def test_chunk_validator_accepts_candidate_modes_and_source_gates(
    tmp_path, chunk_mode, prompt_tokens, chunk_tokens, implementation
):
    tool = load_tool()
    chunk, m1 = write_fixture(
        tmp_path,
        chunk_mode=chunk_mode,
        prompt_tokens=prompt_tokens,
    )

    result = validate(
        tool,
        chunk,
        m1,
        chunk_mode=chunk_mode,
        prompt_tokens=prompt_tokens,
    )

    assert result["passed"] is True
    assert result["prompts"][0]["chunk_vs_source"]["logits"]["top_1_exact"] is True
    assert result["evidence"][0]["schema_version"] == "ullm.sq8.serving_chunks.v4"
    assert result["evidence"][0]["prefill_mode"] == chunk_mode
    assert result["evidence"][0]["prefill_implementation"] == implementation
    chunk_document = json.loads(chunk.read_text(encoding="utf-8"))
    m1_document = json.loads(m1.read_text(encoding="utf-8"))
    assert chunk_document["prefill_chunk_tokens"] == chunk_tokens
    assert m1_document["prefill_chunk_tokens"] == 8
    assert chunk_document["requests"][0]["prefill_execution_units"][0]["width"] == chunk_tokens


def test_candidate_widths_use_selected_chunk_and_m1_tail():
    tool = load_tool()
    assert tool.expected_widths(35, "m32-chunk32") == [32, 1, 1, 1]
    assert tool.expected_widths(131, "m128-chunk128") == [128, 1, 1, 1]
    assert tool.expected_widths(4, "all-m1") == [1, 1, 1, 1]


def test_chunk_validator_rejects_candidate_schema_downgrade(tmp_path):
    tool = load_tool()
    chunk, m1 = write_fixture(tmp_path, chunk_mode="m32-chunk32", prompt_tokens=32)
    document = json.loads(chunk.read_text(encoding="utf-8"))
    document["schema_version"] = "ullm.sq8.serving_chunks.v3"
    chunk.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(tool.ValidationError):
        validate(tool, chunk, m1, chunk_mode="m32-chunk32", prompt_tokens=32)


def test_chunk_validator_cli_preserves_m8_default_and_selects_candidate(monkeypatch):
    tool = load_tool()
    base_args = [
        str(TOOL_PATH),
        "--chunk-result",
        "chunk.json",
        "--all-m1-result",
        "m1.json",
    ]
    monkeypatch.setattr(sys, "argv", base_args)
    assert tool.parse_args().chunk_mode == "m8-chunk8"

    monkeypatch.setattr(sys, "argv", [*base_args, "--chunk-mode", "m128-chunk128"])
    assert tool.parse_args().chunk_mode == "m128-chunk128"


def test_chunk_validator_rejects_cache_trace_tampering(tmp_path):
    tool = load_tool()
    chunk, m1 = write_fixture(tmp_path)
    document = json.loads(chunk.read_text(encoding="utf-8"))
    document["requests"][0]["prefill_execution_units"][0]["cache_lengths"][17] = 7
    chunk.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(tool.ValidationError):
        validate(tool, chunk, m1)


def test_chunk_validator_rejects_capture_alias(tmp_path):
    tool = load_tool()
    chunk, m1 = write_fixture(tmp_path)
    document = json.loads(chunk.read_text(encoding="utf-8"))
    capture = document["requests"][0]["oracle_capture"]
    capture["logits_file"] = "m1-logits.f32le"
    capture["logits_f32_le_sha256"] = sha256_file(tmp_path / "m1-logits.f32le")
    chunk.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(tool.ValidationError):
        validate(tool, chunk, m1)


def test_chunk_validator_rejects_payload_hash_tampering(tmp_path):
    tool = load_tool()
    chunk, m1 = write_fixture(tmp_path)
    payload = tmp_path / "chunk-hidden.f32le"
    values = bytearray(payload.read_bytes())
    values[0] ^= 1
    payload.write_bytes(values)
    with pytest.raises(tool.ValidationError):
        validate(tool, chunk, m1)


def test_chunk_validator_can_require_matching_clean_build_identity(tmp_path):
    tool = load_tool()
    chunk, m1 = write_fixture(tmp_path)
    with pytest.raises(tool.ValidationError):
        tool.validate_results(chunk, m1, FIXTURE_ROOT, (8,), (8,), True)

    for path in (chunk, m1):
        document = json.loads(path.read_text(encoding="utf-8"))
        document["runner_git_commit"] = "a" * 40
        document["runner_worktree_clean"] = True
        document["runner_binary_sha256"] = "b" * 64
        path.write_text(json.dumps(document), encoding="utf-8")
    assert tool.validate_results(chunk, m1, FIXTURE_ROOT, (8,), (8,), True)["passed"]

    document = json.loads(m1.read_text(encoding="utf-8"))
    document["runner_binary_sha256"] = "c" * 64
    m1.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(tool.ValidationError):
        tool.validate_results(chunk, m1, FIXTURE_ROOT, (8,), (8,), True)
