from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/generate-served-model.py"
RECEIPT_TOOL_PATH = ROOT / "tools/write-aq4-resident-promotion-receipt.py"
AQ4_DEPLOYMENT_PROFILE = ROOT / "deploy/served-models/qwen35-9b-aq4.profile.json"
AQ4_REASONING_DEPLOYMENT_PROFILE = (
    ROOT / "deploy/served-models/qwen35-9b-aq4-reasoning.profile.json"
)
EPHEMERAL_COMMIT = "a" * 40


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


GENERATOR = load_module("test_generate_served_model_tool", TOOL_PATH)
RECEIPT_TOOL = load_module("test_write_aq4_promotion_receipt_tool", RECEIPT_TOOL_PATH)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_aq4_split_profile_uses_versioned_receipt_and_production_guard() -> None:
    profile = json.loads(AQ4_DEPLOYMENT_PROFILE.read_text(encoding="utf-8"))
    product_root = Path(profile["product"]["root"])
    receipt = Path(profile["promotion"]["receipt"])
    required_environment = profile["worker"]["required_environment"]

    assert receipt == product_root / "promotion-paged-decode-split-v1.json"
    assert receipt != product_root / "promotion.json"
    assert required_environment.count("ULLM_REQUIRE_HIP_PAGED_DECODE_SPLIT_KERNEL") == 1
    assert not any(
        name.startswith("ULLM_EXPERIMENTAL_HIP_PAGED_DECODE_SPLIT_")
        for name in required_environment
    )


def test_aq4_reasoning_candidate_binds_v2_worker_separately_from_active_v1() -> None:
    profile = json.loads(
        AQ4_REASONING_DEPLOYMENT_PROFILE.read_text(encoding="utf-8")
    )
    worker = profile["worker"]

    assert worker["protocol"] == "ullm.worker.v2"
    assert profile["reasoning"]["dialect_id"] == "qwen3.5-thinking-v1"
    assert Path(worker["binary"]) == (
        ROOT / "target/reasoning-v2/release/ullm-aq4-worker"
    )
    assert Path(worker["binary"]) != ROOT / "target/release/ullm-aq4-worker"
    assert Path(profile["promotion"]["receipt"]) == Path(
        "/home/homelab1/datapool/ullm/product/qwen35-9b-aq4-cli-v0.1/"
        "promotion-reasoning-v2-v0.1.json"
    )
    assert Path(profile["promotion"]["receipt"]) != Path(
        "/home/homelab1/datapool/ullm/product/qwen35-9b-aq4-cli-v0.1/"
        "promotion-paged-decode-split-v1.json"
    )


def write_profile(root: Path, *, receipt_exists: bool = True) -> Path:
    tokenizer = root / "tokenizer"
    tokenizer.mkdir()
    tokenizer_json = b'{"model":"fixture"}\n'
    tokenizer_config = {
        "chat_template": "{{ messages }}",
        "tokenizer_class": "Qwen2Tokenizer",
    }
    (tokenizer / "tokenizer.json").write_bytes(tokenizer_json)
    (tokenizer / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config), encoding="utf-8"
    )

    worker = root / "worker"
    worker.write_bytes(b"#!/bin/sh\nexit 0\n")
    worker.chmod(0o755)

    product = root / "product"
    (product / "artifact").mkdir(parents=True)
    (product / "package").mkdir()
    artifact_manifest = b'{"artifact":true}\n'
    package_manifest = b'{"package":true}\n'
    (product / "artifact/sq_manifest.json").write_bytes(artifact_manifest)
    (product / "package/manifest.json").write_bytes(package_manifest)
    receipt = root / "promotion.json"
    if receipt_exists:
        receipt.write_text(
            json.dumps(
                {
                    "plan_commit": "abc1234",
                    "artifact": {"content_sha256": "1" * 64},
                }
            ),
            encoding="utf-8",
        )

    profile = {
        "schema_version": "ullm.served_model.profile.v1",
        "public": {
            "id": "fixture-model",
            "name": "Fixture model",
            "description": "Generator fixture.",
            "upstream_id": "fixture/upstream",
            "revision": "fixture-revision",
            "context_length": 128,
        },
        "generation": {
            "max_completion_tokens": 16,
            "vocab_size": 100,
            "eos_token_ids": [2],
            "sampling": {"top_k": 1, "temperature": False, "top_p": False},
        },
        "format": {"format_id": "SQ8_0", "implementation_id": "fixture-v1"},
        "tokenizer": {
            "root": os.fspath(tokenizer),
            "transformers_version": "5.12.1",
            "class": "Qwen2Tokenizer",
            "files": ["tokenizer.json", "tokenizer_config.json"],
            "template_options": {
                "add_generation_prompt": True,
                "enable_thinking": False,
            },
        },
        "worker": {
            "protocol": "ullm.worker.v1",
            "binary": os.fspath(worker),
            "arguments": ["--served-model-manifest", "{manifest}"],
            "required_environment": [],
            "identity": {"device": "cpu", "execution_profile": "fixture"},
        },
        "product": {
            "root": os.fspath(product),
            "artifact": {
                "manifest_path": "artifact/sq_manifest.json",
                "content_sha256_from_receipt": ["artifact", "content_sha256"],
            },
            "package": {"manifest_path": "package/manifest.json"},
        },
        "promotion": {
            "receipt": os.fspath(receipt),
            "source_commit_from_receipt": ["plan_commit"],
        },
    }
    path = root / "profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    return path


def write_sq8_v2_ephemeral_profile(root: Path) -> Path:
    profile_path = write_profile(root)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["worker"]["protocol"] = "ullm.worker.v2"
    profile["reasoning"] = {
        "enabled_by_default": False,
        "dialect_id": "synthetic.single-token.v1",
        "start_token_ids": [10],
        "end_token_ids": [20],
        "forced_end_token_ids": [20],
        "initial_phase": "reasoning",
        "eos_policy": "close",
        "effort_budgets": {"low": 2, "medium": 4, "high": 8},
        "max_budget_tokens": 8,
        "reserved_answer_tokens": 1,
        "history_reasoning_policy": "omit",
    }
    receipt_path = root / "promotion.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema_version": GENERATOR.SQ8_EPHEMERAL_RECEIPT_SCHEMA,
                "source_commit": EPHEMERAL_COMMIT,
                "product": {"artifact_content_sha256": "1" * 64},
            }
        ),
        encoding="utf-8",
    )
    profile["promotion"] = {
        "receipt": os.fspath(receipt_path),
        "source_commit_from_receipt": ["source_commit"],
        "required_schema_version": GENERATOR.SQ8_EPHEMERAL_RECEIPT_SCHEMA,
    }
    profile["product"]["artifact"]["content_sha256_from_receipt"] = [
        "product",
        "artifact_content_sha256",
    ]
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    return profile_path


def write_gemma4_e2b_profile(root: Path) -> Path:
    profile_path = write_profile(root)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["public"] = {
        "id": "ullm-gemma4-e2b-bf16",
        "name": "uLLM Gemma4 E2B BF16",
        "description": "Gemma4 E2B text decoder served locally by uLLM.",
        "upstream_id": "google/gemma-4-E2B",
        "revision": "d29ff6b45f081a49ee2733a859c9c9c2d95d1a6f",
        "context_length": 4096,
    }
    profile["generation"] = {
        "max_completion_tokens": 512,
        "vocab_size": 262144,
        "eos_token_ids": [1],
        "sampling": {"top_k": 1, "temperature": False, "top_p": False},
    }
    profile["format"] = {
        "format_id": "BF16_0",
        "implementation_id": "gemma4_e2b_bf16_rdna4_v1",
    }
    profile["tokenizer"]["class"] = "GemmaTokenizer"
    tokenizer_config_path = root / "tokenizer/tokenizer_config.json"
    tokenizer_config_path.write_text(
        json.dumps(
            {
                "chat_template": "{{ messages }}",
                "tokenizer_class": "GemmaTokenizer",
            }
        ),
        encoding="utf-8",
    )
    profile["worker"] = {
        "protocol": "ullm.worker.v1",
        "binary": os.fspath(root / "worker"),
        "arguments": ["--served-model-manifest", "{manifest}"],
        "required_environment": [
            "ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL",
            "ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL",
            "ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL",
        ],
        "identity": {
            "device": "gfx1201",
            "execution_profile": "rdna4_gemma4_e2b_bf16_resident",
        },
    }
    profile["product"]["artifact"] = None
    receipt_path = root / "promotion.json"
    receipt = {
        "schema_version": GENERATOR.GEMMA4_E2B_SERVING_RECEIPT_SCHEMA,
        "source_commit": EPHEMERAL_COMMIT,
        "worker_binary_sha256": sha256((root / "worker").read_bytes()),
        "package_manifest_sha256": sha256(
            (root / "product/package/manifest.json").read_bytes()
        ),
        "tokenizer_chat_template_sha256": sha256(b"{{ messages }}"),
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    profile["promotion"] = {
        "receipt": os.fspath(receipt_path),
        "source_commit_from_receipt": ["source_commit"],
        "required_schema_version": GENERATOR.GEMMA4_E2B_SERVING_RECEIPT_SCHEMA,
    }
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    return profile_path


def test_generate_hashes_live_files_and_passes_strict_loader(tmp_path: Path) -> None:
    profile = write_profile(tmp_path)
    output = tmp_path / "served-model.json"

    digest = GENERATOR.generate(profile, output)
    document = json.loads(output.read_text(encoding="utf-8"))
    loaded = GENERATOR._load_validator().load_served_model(output)

    assert digest == sha256(output.read_bytes()) == loaded.manifest_sha256
    assert document["worker"]["binary_sha256"] == sha256(
        (tmp_path / "worker").read_bytes()
    )
    assert document["tokenizer"]["chat_template_sha256"] == sha256(b"{{ messages }}")
    assert document["product"]["artifact"]["content_sha256"] == "1" * 64
    assert document["promotion"]["source_commit"] == "abc1234"
    assert output.stat().st_mode & 0o777 == 0o644


def test_generator_materializes_v2_reasoning_profile(tmp_path: Path) -> None:
    profile_path = write_sq8_v2_ephemeral_profile(tmp_path)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["worker"]["identity"] = {
        "device": "gfx1201",
        "execution_profile": "rdna4_w8a8_block_ck",
    }
    profile["worker"]["required_environment"] = [
        "ULLM_REQUIRE_HIP_PAGED_DECODE_SPLIT_KERNEL"
    ]
    profile["worker"]["execution"] = {
        "paged_decode_attention": {
            "kernel": "gqa_grouped_split",
            "split_tile": 20,
        }
    }
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    output = tmp_path / "served-model-v2.json"

    GENERATOR.generate_sq8_promotion_ephemeral(profile_path, output)
    document = json.loads(output.read_text(encoding="utf-8"))

    assert document["schema_version"] == "ullm.served_model.v2"
    assert document["worker"]["protocol"] == "ullm.worker.v2"
    assert document["worker"]["execution"] == profile["worker"]["execution"]
    assert document["reasoning"]["dialect_id"] == "synthetic.single-token.v1"


@pytest.mark.parametrize(
    "field",
    ["start_token_ids", "end_token_ids", "forced_end_token_ids"],
)
def test_generator_rejects_multi_token_v2_reasoning_sequences(
    tmp_path: Path, field: str
) -> None:
    profile_path = write_sq8_v2_ephemeral_profile(tmp_path)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["reasoning"][field].append(21)
    profile_path = tmp_path / f"profile-v2-invalid-{field}.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    output = tmp_path / "served-model-v2-invalid.json"

    with pytest.raises(RuntimeError, match="exactly one token"):
        GENERATOR.generate_sq8_promotion_ephemeral(profile_path, output)
    assert not output.exists()


def test_normal_generator_rejects_ephemeral_selector(tmp_path: Path) -> None:
    profile_path = write_sq8_v2_ephemeral_profile(tmp_path)

    with pytest.raises(GENERATOR.GenerationError, match="schema/format/protocol"):
        GENERATOR.materialize(profile_path)


def test_normal_generator_rejects_selectorless_sq8_worker_v2(
    tmp_path: Path,
) -> None:
    profile_path = write_sq8_v2_ephemeral_profile(tmp_path)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["promotion"].pop("required_schema_version")
    profile_path.write_text(json.dumps(profile), encoding="utf-8")

    with pytest.raises(GENERATOR.GenerationError, match="schema/format/protocol"):
        GENERATOR.materialize(profile_path)


def test_generator_admits_only_the_bound_gemma4_bf16_receipt(tmp_path: Path) -> None:
    profile_path = write_gemma4_e2b_profile(tmp_path)
    document = GENERATOR.materialize(profile_path)

    assert document["format"]["format_id"] == "BF16_0"
    assert document["worker"]["protocol"] == "ullm.worker.v1"
    assert document["promotion"]["source_commit"] == EPHEMERAL_COMMIT

    receipt_path = tmp_path / "promotion.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["package_manifest_sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(GENERATOR.GenerationError, match="Gemma4 promotion receipt"):
        GENERATOR.materialize(profile_path)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda profile: profile["format"].__setitem__("format_id", "UNKNOWN"),
        lambda profile: profile["worker"].__setitem__(
            "protocol", "ullm.worker.unknown"
        ),
        lambda profile: profile["promotion"].__setitem__(
            "required_schema_version", "ullm.promotion.unknown"
        ),
    ],
)
def test_generator_rejects_unknown_format_protocol_or_schema_pair(
    tmp_path: Path,
    mutate,
) -> None:
    profile_path = write_profile(tmp_path)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    mutate(profile)
    profile_path.write_text(json.dumps(profile), encoding="utf-8")

    with pytest.raises(GENERATOR.GenerationError, match="schema/format/protocol"):
        GENERATOR.materialize(profile_path)


def test_v2_generator_profile_requires_reasoning(tmp_path: Path) -> None:
    profile = json.loads(write_profile(tmp_path).read_text(encoding="utf-8"))
    profile["worker"]["protocol"] = "ullm.worker.v2"
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")

    with pytest.raises(GENERATOR.GenerationError, match="requires reasoning"):
        GENERATOR.materialize(profile_path)


def test_generator_rejects_execution_settings_on_a_v1_profile(tmp_path: Path) -> None:
    profile_path = write_profile(tmp_path)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["worker"]["execution"] = {
        "paged_decode_attention": {
            "kernel": "gqa_grouped_split",
            "split_tile": 20,
        }
    }
    profile_path.write_text(json.dumps(profile), encoding="utf-8")

    with pytest.raises(GENERATOR.GenerationError, match="requires ullm.worker.v2"):
        GENERATOR.materialize(profile_path)


def test_v2_promotion_validator_recomputes_budget_zero_case() -> None:
    manifest = {
        "worker": {"protocol": "ullm.worker.v2"},
        "reasoning": {
            "dialect_id": "synthetic.single-token.v1",
            "start_token_ids": [10],
            "end_token_ids": [20],
            "forced_end_token_ids": [20],
            "reserved_answer_tokens": 1,
        },
    }
    evidence = {
        "resident": {
            "ready": {"schema_version": "ullm.worker.v2"},
            "cases": [
                {"id": "raw-p0001-g0004", "tokens": [30]},
                {
                    "id": "reasoning-budget-zero",
                    "reasoning": {
                        "enabled": True,
                        "budget_tokens": 0,
                        "dialect_id": "synthetic.single-token.v1",
                        "end_token_ids": [20],
                        "forced_end_token_ids": [20],
                        "reserved_answer_tokens": 1,
                    },
                    "reasoning_usage": {
                        "reasoning_tokens": 0,
                        "forced_end_tokens": 1,
                    },
                    "tokens": [20, 30],
                },
            ],
        },
        "legacy": {
            "ready": {"schema_version": "ullm.worker.v1"},
            "cases": [{"id": "raw-p0001-g0004", "tokens": [30]}],
        },
    }

    GENERATOR._validate_v2_reasoning_evidence(evidence, manifest)

    for field in (
        "start_token_ids",
        "end_token_ids",
        "forced_end_token_ids",
    ):
        invalid_manifest = copy.deepcopy(manifest)
        invalid_manifest["reasoning"][field].append(21)
        with pytest.raises(GENERATOR.GenerationError, match="exactly one token"):
            GENERATOR._validate_v2_reasoning_evidence(evidence, invalid_manifest)

    for field in ("end_token_ids", "forced_end_token_ids"):
        invalid_evidence = copy.deepcopy(evidence)
        invalid_evidence["resident"]["cases"][1]["reasoning"][field].append(21)
        with pytest.raises(GENERATOR.GenerationError, match="exactly one token"):
            GENERATOR._validate_v2_reasoning_evidence(invalid_evidence, manifest)

    evidence["resident"]["cases"][1]["tokens"] = [30, 20]
    with pytest.raises(GENERATOR.GenerationError, match="accounting"):
        GENERATOR._validate_v2_reasoning_evidence(evidence, manifest)


def test_missing_promotion_receipt_fails_without_output(tmp_path: Path) -> None:
    profile = write_profile(tmp_path, receipt_exists=False)
    output = tmp_path / "served-model.json"

    with pytest.raises(GENERATOR.GenerationError, match="promotion receipt"):
        GENERATOR.generate(profile, output)

    assert not output.exists()


def test_refuses_symlink_output(tmp_path: Path) -> None:
    profile = write_profile(tmp_path)
    target = tmp_path / "target.json"
    target.write_text("unchanged", encoding="utf-8")
    output = tmp_path / "served-model.json"
    output.symlink_to(target)

    with pytest.raises(GENERATOR.GenerationError, match="must not be a symlink"):
        GENERATOR.generate(profile, output)

    assert target.read_text(encoding="utf-8") == "unchanged"


def write_aq4_profile(root: Path) -> tuple[Path, Path, dict[str, object]]:
    profile_path = write_profile(root)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["format"] = {
        "format_id": "AQ4_0",
        "implementation_id": "qwen35_aq4_rdna4_v1",
    }
    profile["worker"]["identity"] = {
        "device": "gfx1201",
        "execution_profile": "rdna4_aq4_resident",
    }
    profile["worker"]["required_environment"] = [
        "ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL",
        "ULLM_REQUIRE_HIP_AQ4_REGISTER_BM8_KERNEL",
        "ULLM_REQUIRE_HIP_AQ4_REGISTER_BM8_GROUP8_KERNEL",
        "ULLM_REQUIRE_HIP_AQ4_WMMA_GEMM_KERNEL",
        "ULLM_REQUIRE_HIP_AQ4_WMMA_GEMM_GROUP8_KERNEL",
        "ULLM_REQUIRE_HIP_AQ4_WMMA_GEMM_RAGGED_M_KERNEL",
        "ULLM_REQUIRE_HIP_AQ4_WMMA_GEMM_GROUP8_RAGGED_M_KERNEL",
        "ULLM_REQUIRE_HIP_LINEAR_ATTN_QKV_PREPARE_BATCH_KERNEL",
        "ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_SEQUENCE_KERNEL",
        "ULLM_REQUIRE_HIP_PAGED_KV_WRITE_CHUNK_KERNEL",
        "ULLM_REQUIRE_HIP_PAGED_CAUSAL_GQA_CHUNK_KERNEL",
        "ULLM_REQUIRE_HIP_PAGED_CAUSAL_GQA_WMMA_KERNEL",
        "ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_BATCH_KERNEL",
    ]
    profile["product"]["artifact"] = None
    profile["promotion"] = {
        "receipt": os.fspath(root / "promotion.json"),
        "source_commit_from_receipt": ["source_commit"],
        "required_schema_version": GENERATOR.AQ4_EPHEMERAL_RECEIPT_SCHEMA,
    }
    (root / "promotion.json").write_text(
        json.dumps(
            {
                "schema_version": GENERATOR.AQ4_EPHEMERAL_RECEIPT_SCHEMA,
                "source_commit": EPHEMERAL_COMMIT,
            }
        ),
        encoding="utf-8",
    )
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    ephemeral_manifest_path = root / "aq4-ephemeral-served-model.json"
    GENERATOR.generate_aq4_promotion_ephemeral(
        profile_path,
        ephemeral_manifest_path,
    )
    bound_manifest = json.loads(ephemeral_manifest_path.read_text(encoding="utf-8"))

    evidence: dict[str, object] = {
        "schema_version": "ullm.aq4_resident_promotion_evidence.v1",
        "source_commit": EPHEMERAL_COMMIT,
        "production_receipt_written": False,
        "gpu_exclusive_preflight": {
            "tool": "rocm-smi --showpids --json",
            "gpu_index": "1",
            "positive_vram_processes": [],
        },
        "verified": True,
        "worker_binary": bound_manifest["worker"]["binary"],
        "worker_binary_sha256": bound_manifest["worker"]["binary_sha256"],
        "ephemeral_bundle": {"manifest": bound_manifest},
        "resident": {
            "clean_shutdown": True,
            "cases": [{"id": "fixture", "tokens": [1, 2]}],
            "child_process_checks": [{"sibling_engine_count": 0}],
        },
        "legacy": {"clean_shutdown": True, "cases": [{"id": "fixture", "tokens": [1, 2]}]},
        "comparisons": [{"id": "fixture", "tokens_exact_match": True}],
    }
    evidence_path = root / "resident-evidence.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    receipt = {
        "schema_version": "ullm.aq4_resident_promotion.v1",
        "source_commit": EPHEMERAL_COMMIT,
        "evidence": {
            "path": evidence_path.name,
            "sha256": sha256(evidence_path.read_bytes()),
        },
    }
    (root / "promotion.json").write_text(json.dumps(receipt), encoding="utf-8")
    profile["promotion"].update(
        {
            "required_schema_version": "ullm.aq4_resident_promotion.v1",
            "evidence_from_receipt": ["evidence", "path"],
            "evidence_sha256_from_receipt": ["evidence", "sha256"],
        }
    )
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    return profile_path, evidence_path, evidence


def rewrite_aq4_evidence(root: Path, evidence_path: Path, evidence: dict[str, object]) -> None:
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    receipt_path = root / "promotion.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["evidence"]["sha256"] = sha256(evidence_path.read_bytes())
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")


def test_aq4_evidence_gate_accepts_fully_bound_verified_evidence(tmp_path: Path) -> None:
    profile, _, _ = write_aq4_profile(tmp_path)

    document = GENERATOR.materialize(profile)

    assert document["format"]["format_id"] == "AQ4_0"
    assert document["promotion"]["source_commit"] == EPHEMERAL_COMMIT
    assert document["worker"]["required_environment"] == [
        "ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL",
        "ULLM_REQUIRE_HIP_AQ4_REGISTER_BM8_KERNEL",
        "ULLM_REQUIRE_HIP_AQ4_REGISTER_BM8_GROUP8_KERNEL",
        "ULLM_REQUIRE_HIP_AQ4_WMMA_GEMM_KERNEL",
        "ULLM_REQUIRE_HIP_AQ4_WMMA_GEMM_GROUP8_KERNEL",
        "ULLM_REQUIRE_HIP_AQ4_WMMA_GEMM_RAGGED_M_KERNEL",
        "ULLM_REQUIRE_HIP_AQ4_WMMA_GEMM_GROUP8_RAGGED_M_KERNEL",
        "ULLM_REQUIRE_HIP_LINEAR_ATTN_QKV_PREPARE_BATCH_KERNEL",
        "ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_SEQUENCE_KERNEL",
        "ULLM_REQUIRE_HIP_PAGED_KV_WRITE_CHUNK_KERNEL",
        "ULLM_REQUIRE_HIP_PAGED_CAUSAL_GQA_CHUNK_KERNEL",
        "ULLM_REQUIRE_HIP_PAGED_CAUSAL_GQA_WMMA_KERNEL",
        "ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_BATCH_KERNEL",
    ]


@pytest.mark.parametrize("worker_protocol", ["ullm.worker.v1", "ullm.worker.v2"])
def test_aq4_profile_requires_promotion_selector(
    tmp_path: Path, worker_protocol: str
) -> None:
    profile_path, _, _ = write_aq4_profile(tmp_path)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["promotion"].pop("required_schema_version")
    if worker_protocol == "ullm.worker.v2":
        profile["worker"]["protocol"] = worker_protocol
        profile["reasoning"] = {
            "enabled_by_default": False,
            "dialect_id": "synthetic.single-token.v1",
            "start_token_ids": [10],
            "end_token_ids": [20],
            "forced_end_token_ids": [20],
            "initial_phase": "reasoning",
            "eos_policy": "close",
            "effort_budgets": {"low": 2, "medium": 4, "high": 8},
            "max_budget_tokens": 8,
            "reserved_answer_tokens": 1,
            "history_reasoning_policy": "omit",
        }
    profile_path.write_text(json.dumps(profile), encoding="utf-8")

    with pytest.raises(GENERATOR.GenerationError, match="schema/format/protocol"):
        GENERATOR.materialize(profile_path)


def test_normal_generator_rejects_aq4_ephemeral_scaffold(tmp_path: Path) -> None:
    profile_path, _, _ = write_aq4_profile(tmp_path)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    scaffold_path = tmp_path / "aq4-normal-path-scaffold.json"
    scaffold_path.write_text(
        json.dumps(
            {
                "schema_version": GENERATOR.AQ4_EPHEMERAL_RECEIPT_SCHEMA,
                "source_commit": EPHEMERAL_COMMIT,
            }
        ),
        encoding="utf-8",
    )
    profile["promotion"] = {
        "receipt": os.fspath(scaffold_path),
        "source_commit_from_receipt": ["source_commit"],
        "required_schema_version": GENERATOR.AQ4_EPHEMERAL_RECEIPT_SCHEMA,
    }
    profile_path.write_text(json.dumps(profile), encoding="utf-8")

    with pytest.raises(GENERATOR.GenerationError, match="schema/format/protocol"):
        GENERATOR.materialize(profile_path)


def test_aq4_ephemeral_generator_requires_full_lowercase_commit(
    tmp_path: Path,
) -> None:
    profile_path, _, _ = write_aq4_profile(tmp_path)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    scaffold_path = tmp_path / "aq4-invalid-commit-scaffold.json"
    scaffold_path.write_text(
        json.dumps(
            {
                "schema_version": GENERATOR.AQ4_EPHEMERAL_RECEIPT_SCHEMA,
                "source_commit": "A" * 40,
            }
        ),
        encoding="utf-8",
    )
    profile["promotion"] = {
        "receipt": os.fspath(scaffold_path),
        "source_commit_from_receipt": ["source_commit"],
        "required_schema_version": GENERATOR.AQ4_EPHEMERAL_RECEIPT_SCHEMA,
    }
    profile_path.write_text(json.dumps(profile), encoding="utf-8")

    with pytest.raises(GENERATOR.GenerationError, match="ephemeral"):
        GENERATOR.generate_aq4_promotion_ephemeral(
            profile_path,
            tmp_path / "aq4-invalid-commit-manifest.json",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.__setitem__("verified", False), "not verified"),
        (
            lambda value: value.__setitem__("production_receipt_written", True),
            "before receipt publication",
        ),
        (
            lambda value: value.pop("gpu_exclusive_preflight"),
            "GPU exclusivity preflight",
        ),
        (
            lambda value: value["gpu_exclusive_preflight"].__setitem__(
                "positive_vram_processes", [{"pid": "42"}]
            ),
            "GPU exclusivity preflight",
        ),
        (lambda value: value.__setitem__("source_commit", "other"), "source commit"),
        (
            lambda value: value["comparisons"][0].__setitem__("tokens_exact_match", False),
            "comparisons",
        ),
        (
            lambda value: value["resident"]["cases"][0].__setitem__("tokens", [99]),
            "token comparisons",
        ),
        (
            lambda value: value["resident"].__setitem__("clean_shutdown", False),
            "resident shutdown",
        ),
        (
            lambda value: value["legacy"].__setitem__("clean_shutdown", False),
            "legacy shutdown",
        ),
        (
            lambda value: value["resident"]["child_process_checks"][0].__setitem__(
                "sibling_engine_count", 1
            ),
            "child-process",
        ),
        (
            lambda value: value["ephemeral_bundle"]["manifest"]["worker"][
                "identity"
            ].__setitem__("execution_profile", "legacy"),
            "worker identity",
        ),
        (
            lambda value: value["ephemeral_bundle"]["manifest"]["product"][
                "package"
            ].__setitem__("manifest_sha256", "0" * 64),
            "package identity",
        ),
        (
            lambda value: value["ephemeral_bundle"]["manifest"]["public"].__setitem__(
                "revision", "other"
            ),
            "profile public",
        ),
    ],
)
def test_aq4_evidence_gate_fails_closed(
    tmp_path: Path, mutation: object, message: str
) -> None:
    profile, evidence_path, evidence = write_aq4_profile(tmp_path)
    mutation(evidence)  # type: ignore[operator]
    rewrite_aq4_evidence(tmp_path, evidence_path, evidence)

    with pytest.raises(GENERATOR.GenerationError, match=message):
        GENERATOR.materialize(profile)


def test_aq4_evidence_hash_and_safe_relative_path_are_required(tmp_path: Path) -> None:
    profile, evidence_path, _ = write_aq4_profile(tmp_path)
    receipt_path = tmp_path / "promotion.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["evidence"]["sha256"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(GENERATOR.GenerationError, match="SHA-256 differs"):
        GENERATOR.materialize(profile)

    receipt["evidence"]["path"] = "../resident-evidence.json"
    receipt["evidence"]["sha256"] = sha256(evidence_path.read_bytes())
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(GENERATOR.GenerationError, match="safe relative path"):
        GENERATOR.materialize(profile)


def test_aq4_evidence_symlink_is_rejected(tmp_path: Path) -> None:
    profile, evidence_path, _ = write_aq4_profile(tmp_path)
    link = tmp_path / "linked-evidence.json"
    link.symlink_to(evidence_path)
    receipt_path = tmp_path / "promotion.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["evidence"]["path"] = link.name
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(GENERATOR.GenerationError, match="non-symlink"):
        GENERATOR.materialize(profile)


def test_receipt_tool_validates_then_atomically_publishes(tmp_path: Path) -> None:
    profile, evidence_path, _ = write_aq4_profile(tmp_path)
    receipt_path = tmp_path / "promotion.json"
    receipt_path.unlink()

    receipt = RECEIPT_TOOL.write_receipt(profile, evidence_path, receipt_path)

    assert json.loads(receipt_path.read_text(encoding="ascii")) == receipt
    assert receipt["schema_version"] == "ullm.aq4_resident_promotion.v1"
    assert receipt["evidence"]["path"] == evidence_path.name
    assert (
        GENERATOR.materialize(profile)["promotion"]["source_commit"]
        == EPHEMERAL_COMMIT
    )
    with pytest.raises(RECEIPT_TOOL.ReceiptError, match="already exists"):
        RECEIPT_TOOL.write_receipt(profile, evidence_path, receipt_path)


def test_receipt_tool_does_not_publish_invalid_evidence(tmp_path: Path) -> None:
    profile, evidence_path, evidence = write_aq4_profile(tmp_path)
    receipt_path = tmp_path / "promotion.json"
    receipt_path.unlink()
    evidence["verified"] = False
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    with pytest.raises(RECEIPT_TOOL.ReceiptError, match="not verified"):
        RECEIPT_TOOL.write_receipt(profile, evidence_path, receipt_path)

    assert not receipt_path.exists()
