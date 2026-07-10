#!/usr/bin/env python3
"""Validate the fixed Qwen3-14B-FP8 M=8 greedy eight-step oracle."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA_VERSION = "ullm.qwen3_generation_oracle.v1"
TRUSTED_METADATA_SHA256 = "5fc03a28cd15409e84a7fd23fd51c0cbd6ec9cf8761a66d1f5ede7ddfe3226a0"
PROMPT_TOKEN_IDS = list(range(1, 9))
PROMPT_POSITION_IDS = list(range(8))
GENERATED_TOKEN_IDS = [353, 10, 4999, 1725, 15, 16, 17, 18]
STEPS = 8
TOP_K = 10
HIDDEN_SIZE = 5120
VOCAB_SIZE = 151936
EXPECTED_REVISION = "9a283b4a5efbc09ce247e0ae5b02b744739e525a"
EXPECTED_REVISION_FILES = {
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
}
EXPECTED_CHECKPOINT_FILES = [
    {
        "file": "config.json",
        "bytes": 896,
        "sha256": "c5d7d0e8ee42088bd535101d13c71d38c20b5c2afd46ee8fdfba351956233793",
    },
    {
        "file": "model.safetensors.index.json",
        "bytes": 62044,
        "sha256": "6a9c8e17744118347080916d8f673b881941cf42989ee77266b14dc2062a7151",
    },
    {
        "file": "model-00001-of-00004.safetensors",
        "bytes": 4922397616,
        "sha256": "2c2f93f7639950a7246c54457482696b94aa0e6b1f49d2169f0422f56c1ed370",
    },
    {
        "file": "model-00002-of-00004.safetensors",
        "bytes": 4955472248,
        "sha256": "7831581bc7d03d77707df3ef10b8d90ee1998ee890ea0020b4a62d27079925ba",
    },
    {
        "file": "model-00003-of-00004.safetensors",
        "bytes": 4892558664,
        "sha256": "d57d1788fb339440b12c6917f7f88e18a5cb76e20f0bfacadd9e4e70a49b2a2a",
    },
    {
        "file": "model-00004-of-00004.safetensors",
        "bytes": 1555824768,
        "sha256": "b4bf668aa6f8535dd467a9a3339116b536682b4241972054b783d514cbe84e50",
    },
    {
        "file": "tokenizer_config.json",
        "bytes": 9732,
        "sha256": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    },
]
EXPECTED_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 5120,
    "initializer_range": 0.02,
    "intermediate_size": 17408,
    "max_position_embeddings": 40960,
    "max_window_layers": 40,
    "model_type": "qwen3",
    "num_attention_heads": 40,
    "num_hidden_layers": 40,
    "num_key_value_heads": 8,
    "quantization_config": {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
    },
    "rms_norm_eps": 1e-6,
    "rope_scaling": None,
    "rope_theta": 1_000_000,
    "sliding_window": None,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.51.0",
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 151936,
}
TRUSTED_ENVIRONMENT = {
    "gpu": {
        "compute_capability": [12, 0],
        "gfx": "gfx1201",
        "name": "AMD Radeon Graphics",
        "rocr_visible_devices": "1",
        "total_memory_bytes": 34208743424,
        "visible_device_index": 0,
    },
    "packages": {
        "accelerate": "1.14.0",
        "numpy": "2.1.3",
        "safetensors": "0.8.0",
        "torch": "2.11.0+gitd0c8b1f",
        "transformers": "5.12.1",
        "triton": "3.6.0",
        "vllm": "0.23.1rc1.dev618+g8cf7c4d8a.rocm723",
    },
    "platform": "Linux-6.17.0-35-generic-x86_64-with-glibc2.39",
    "python": "3.12.3 (main, Jun 19 2026, 12:46:00) [GCC 13.3.0]",
    "python_executable": (
        "/home/homelab1/coding-local/ultimateLLM/"
        "uLLM-project/build/envs/vllm-rocm-nightly/bin/python"
    ),
    "rocm_version_file": "7.2.1",
    "torch_git_version": "d0c8b1f364ecacff4dd8bc06a645d0fb9324cd37",
    "torch_hip_version": "7.2.53211",
}
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class ValidationError(ValueError):
    pass


def fail(message: str) -> None:
    raise ValidationError(message)


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_metadata(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        fail(f"failed to load metadata.json: {error}")
    if not isinstance(value, dict):
        fail("metadata.json must contain an object")
    return value


def exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail(f"{label} must be an object")
    if set(value) != expected:
        fail(
            f"{label} keys differ: missing={sorted(expected - set(value))} "
            f"extra={sorted(set(value) - expected)}"
        )
    return value


def exact_list(value: Any, length: int, label: str) -> list[Any]:
    if not isinstance(value, list) or len(value) != length:
        fail(f"{label} must contain exactly {length} entries")
    return value


def integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        fail(f"{label} must be an integer")
    return value


def finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        fail(f"{label} must be finite")
    return result


def valid_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        fail(f"{label} must be a lowercase SHA-256 digest")
    return value


def sha256_file(path: Path, chunk_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def safe_artifact_path(root: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        fail(f"{label} must be a non-empty relative path")
    pure = PurePosixPath(raw)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        fail(f"{label} is unsafe: {raw!r}")
    path = root.joinpath(*pure.parts)
    try:
        info = path.lstat()
    except OSError as error:
        fail(f"missing artifact {raw}: {error}")
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        fail(f"artifact must be a regular non-symlink file: {raw}")
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError):
        fail(f"artifact escapes oracle directory: {raw}")
    return path


def validate_created_utc(value: Any) -> None:
    if not isinstance(value, str):
        fail("created_utc must be an ISO-8601 string")
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError as error:
        fail(f"created_utc is invalid: {error}")
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        fail("created_utc must include an explicit UTC offset")


def validate_source(value: Any) -> None:
    source = exact_keys(
        value,
        {"name", "local_dir", "revision", "checkpoint_files", "config"},
        "source",
    )
    if source["name"] != "Qwen/Qwen3-14B-FP8":
        fail("source.name is invalid")
    if not isinstance(source["local_dir"], str) or not Path(source["local_dir"]).is_absolute():
        fail("source.local_dir must be absolute")
    if source["checkpoint_files"] != EXPECTED_CHECKPOINT_FILES:
        fail("source.checkpoint_files differs from the fixed checkpoint")
    if source["config"] != EXPECTED_CONFIG:
        fail("source.config differs from the fixed model config")
    revision = exact_keys(
        source["revision"],
        {"revision", "revision_consistent", "per_file_revisions"},
        "source.revision",
    )
    per_file = revision["per_file_revisions"]
    if (
        revision["revision"] != EXPECTED_REVISION
        or revision["revision_consistent"] is not True
        or not isinstance(per_file, dict)
        or set(per_file) != EXPECTED_REVISION_FILES
        or any(item != EXPECTED_REVISION for item in per_file.values())
    ):
        fail("source revision set differs from the fixed checkpoint")


def validate_prompt(value: Any) -> None:
    expected = {
        "token_ids": PROMPT_TOKEN_IDS,
        "position_ids": PROMPT_POSITION_IDS,
        "attention": "causal",
        "bos_inserted": False,
        "chat_template_applied": False,
    }
    if value != expected:
        fail("prompt differs from the fixed raw-token M=8 contract")


def validate_generation(value: Any, generated: list[int]) -> None:
    generation = exact_keys(
        value,
        {
            "method",
            "temperature",
            "max_new_tokens",
            "min_new_tokens",
            "fixed_step_count",
            "ignore_eos",
            "early_stop_on_eos",
            "eos_token_id",
            "eos_generated_at_steps",
            "finish_reason",
            "top_k_recorded",
            "topk_tie_breaker",
            "seed",
        },
        "generation",
    )
    expected_eos_steps = [
        index for index, token_id in enumerate(generated) if token_id == 151645
    ]
    expected = {
        "method": "greedy",
        "temperature": 0.0,
        "max_new_tokens": STEPS,
        "min_new_tokens": STEPS,
        "fixed_step_count": STEPS,
        "ignore_eos": True,
        "early_stop_on_eos": False,
        "eos_token_id": 151645,
        "eos_generated_at_steps": expected_eos_steps,
        "finish_reason": "length",
        "top_k_recorded": TOP_K,
        "topk_tie_breaker": "token_id_ascending",
        "seed": 0,
    }
    if generation != expected:
        fail("generation differs from the fixed greedy eight-step contract")


def validate_model_info(value: Any) -> None:
    entries = exact_list(value, 1, "execution.model_info")
    expected = {
        "model_class": "Qwen3ForCausalLM",
        "decoder_layer_class": "Qwen3DecoderLayer",
        "decoder_layer_count": 40,
        "final_norm_class": "RMSNorm",
        "lm_head_class": "ParallelLMHead",
        "quant_config_class": "Fp8Config",
        "qkv_quant_method_class": "Fp8LinearMethod",
        "hidden_size": HIDDEN_SIZE,
        "vocab_size": VOCAB_SIZE,
        "tie_word_embeddings": False,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000.0,
        "head_dim": 128,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
    }
    if entries[0] != expected:
        fail("execution.model_info differs from the fixed Qwen3/vLLM model identity")


def validate_execution(value: Any) -> None:
    execution = exact_keys(
        value,
        {
            "backend",
            "runner",
            "dtype",
            "quantization",
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "max_model_len",
            "max_num_seqs",
            "max_num_batched_tokens",
            "kv_cache_memory_bytes",
            "enforce_eager",
            "enable_prefix_caching",
            "async_scheduling",
            "v1_multiprocessing",
            "forward_token_counts",
            "model_info",
        },
        "execution",
    )
    expected = {
        "backend": "vLLM",
        "runner": "generate",
        "dtype": "bfloat16",
        "quantization": EXPECTED_CONFIG["quantization_config"],
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "max_model_len": 16,
        "max_num_seqs": 1,
        "max_num_batched_tokens": 8,
        "kv_cache_memory_bytes": 64 * 1024 * 1024,
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "async_scheduling": False,
        "v1_multiprocessing": False,
        "forward_token_counts": [8] + [1] * 7,
    }
    for key, expected_value in expected.items():
        if execution[key] != expected_value:
            fail(f"execution.{key} differs from the fixed contract")
    validate_model_info(execution["model_info"])


def validate_environment(value: Any, trusted: bool) -> None:
    environment = exact_keys(
        value,
        {
            "python",
            "python_executable",
            "platform",
            "packages",
            "torch_git_version",
            "torch_hip_version",
            "rocm_version_file",
            "gpu",
        },
        "environment",
    )
    packages = exact_keys(
        environment["packages"],
        {"vllm", "torch", "transformers", "safetensors", "accelerate", "triton", "numpy"},
        "environment.packages",
    )
    if any(not isinstance(item, str) or not item for item in packages.values()):
        fail("environment package versions must be non-empty strings")
    gpu = exact_keys(
        environment["gpu"],
        {
            "visible_device_index",
            "name",
            "gfx",
            "total_memory_bytes",
            "compute_capability",
            "rocr_visible_devices",
        },
        "environment.gpu",
    )
    if (
        gpu["visible_device_index"] != 0
        or gpu["gfx"] != "gfx1201"
        or gpu["compute_capability"] != [12, 0]
        or gpu["rocr_visible_devices"] != "1"
        or integer(gpu["total_memory_bytes"], "environment.gpu.total_memory_bytes")
        < 32_000_000_000
    ):
        fail("environment.gpu is not the isolated R9700/gfx1201 contract")
    for key in ("python", "python_executable", "platform", "torch_git_version", "torch_hip_version"):
        if not isinstance(environment[key], str) or not environment[key]:
            fail(f"environment.{key} must be a non-empty string")
    if trusted and environment != TRUSTED_ENVIRONMENT:
        fail("trusted environment differs from the frozen vLLM/ROCm identity")


def validate_artifact_manifest(
    root: Path, value: Any
) -> dict[str, dict[str, Any]]:
    records = exact_list(value, 18, "artifact_files_excluding_metadata")
    parsed: dict[str, dict[str, Any]] = {}
    ordered = []
    for index, raw_record in enumerate(records):
        record = exact_keys(raw_record, {"file", "bytes", "sha256"}, f"artifact[{index}]")
        name = record["file"]
        path = safe_artifact_path(root, name, f"artifact[{index}].file")
        if name in parsed:
            fail(f"duplicate artifact path: {name}")
        ordered.append(name)
        if integer(record["bytes"], f"artifact[{index}].bytes") != path.stat().st_size:
            fail(f"artifact byte count mismatch: {name}")
        digest = valid_sha256(record["sha256"], f"artifact[{index}].sha256")
        if sha256_file(path) != digest:
            fail(f"artifact SHA-256 mismatch: {name}")
        parsed[name] = record
    if ordered != sorted(ordered):
        fail("artifact manifest must be sorted by path")
    required = {"export_generation_oracle.py", "rerun-command.sh"}
    for step in range(STEPS):
        required.add(f"steps/step-{step:02d}-final-hidden.f32")
        required.add(f"steps/step-{step:02d}-logits.f32")
    if set(parsed) != required:
        fail("artifact manifest has missing or unexpected files")
    actual_files = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            fail(f"oracle directory contains a symlink: {relative}")
        if stat.S_ISREG(info.st_mode) and relative != "metadata.json":
            actual_files.add(relative)
        elif stat.S_ISDIR(info.st_mode):
            if relative != "steps":
                fail(f"oracle directory contains an unexpected directory: {relative}")
        elif relative != "metadata.json":
            fail(f"oracle directory contains a non-regular entry: {relative}")
    if actual_files != set(parsed):
        fail("artifact manifest does not cover every file excluding metadata.json")
    return parsed


def compute_health(path: Path, expected_elements: int) -> dict[str, Any]:
    try:
        import numpy as np
        import torch
    except ImportError as error:
        fail(f"validator requires numpy and torch: {error}")
    values = np.fromfile(path, dtype="<f4")
    if int(values.size) != expected_elements:
        fail(f"decoded element count mismatch for {path.name}")
    tensor = torch.from_numpy(values)
    finite = torch.isfinite(tensor)
    finite_count = int(finite.sum().item())
    if finite_count != expected_elements:
        fail(f"{path.name} contains non-finite values")
    finite_values = tensor[finite]
    return {
        "elements": int(tensor.numel()),
        "finite_count": finite_count,
        "nan_count": int(torch.isnan(tensor).sum().item()),
        "inf_count": int(torch.isinf(tensor).sum().item()),
        "min": float(finite_values.min().item()),
        "max": float(finite_values.max().item()),
        "mean": float(finite_values.mean().item()),
        "std_population": float(finite_values.std(unbiased=False).item()),
        "l2": float(torch.linalg.vector_norm(finite_values).item()),
        "max_abs": float(finite_values.abs().max().item()),
    }


def validate_health(reported: Any, actual: dict[str, Any], label: str) -> None:
    health = exact_keys(
        reported,
        {
            "elements",
            "finite_count",
            "nan_count",
            "inf_count",
            "min",
            "max",
            "mean",
            "std_population",
            "l2",
            "max_abs",
        },
        f"{label}.health",
    )
    for key in ("elements", "finite_count", "nan_count", "inf_count"):
        if integer(health[key], f"{label}.health.{key}") != actual[key]:
            fail(f"{label}.health.{key} does not match the payload")
    if actual["finite_count"] != actual["elements"] or actual["nan_count"] or actual["inf_count"]:
        fail(f"{label} contains non-finite values")
    for key in ("min", "max", "mean", "std_population", "l2", "max_abs"):
        value = finite_number(health[key], f"{label}.health.{key}")
        if not math.isclose(value, actual[key], rel_tol=1e-7, abs_tol=1e-7):
            fail(f"{label}.health.{key} does not match the payload")


def validate_tensor(
    root: Path,
    manifests: dict[str, dict[str, Any]],
    value: Any,
    expected_file: str,
    elements: int,
    label: str,
) -> Path:
    tensor = exact_keys(
        value,
        {"file", "shape", "storage_dtype", "source_dtype", "bytes", "sha256", "health"},
        label,
    )
    expected_bytes = elements * 4
    if (
        tensor["file"] != expected_file
        or tensor["shape"] != [elements]
        or tensor["storage_dtype"] != "float32_le"
        or tensor["source_dtype"] != "torch.bfloat16"
        or integer(tensor["bytes"], f"{label}.bytes") != expected_bytes
    ):
        fail(f"{label} identity, shape, dtype, or byte count is invalid")
    digest = valid_sha256(tensor["sha256"], f"{label}.sha256")
    manifest = manifests.get(expected_file)
    if manifest is None or manifest["bytes"] != expected_bytes or manifest["sha256"] != digest:
        fail(f"{label} does not match the artifact manifest")
    path = safe_artifact_path(root, expected_file, f"{label}.file")
    validate_health(tensor["health"], compute_health(path, elements), label)
    return path


def recompute_top_10(logits_path: Path) -> list[dict[str, Any]]:
    try:
        import numpy as np
    except ImportError as error:
        fail(f"validator requires numpy for top-k: {error}")
    logits = np.fromfile(logits_path, dtype="<f4")
    if int(logits.size) != VOCAB_SIZE:
        fail("logits payload has the wrong shape")
    token_ids = np.arange(VOCAB_SIZE, dtype=np.int64)
    indices = np.lexsort((token_ids, -logits))[:TOP_K]
    return [
        {"token_id": int(token_id), "logit": float(logits[token_id])}
        for token_id in indices
    ]


def validate_top_10(reported: Any, actual: list[dict[str, Any]], label: str) -> None:
    entries = exact_list(reported, TOP_K, label)
    for rank, (entry, expected) in enumerate(zip(entries, actual, strict=True)):
        parsed = exact_keys(entry, {"token_id", "logit"}, f"{label}[{rank}]")
        if integer(parsed["token_id"], f"{label}[{rank}].token_id") != expected["token_id"]:
            fail(f"{label} token IDs do not match logits")
        if finite_number(parsed["logit"], f"{label}[{rank}].logit") != expected["logit"]:
            fail(f"{label} values do not match logits")


def validate_steps(
    root: Path,
    manifests: dict[str, dict[str, Any]],
    value: Any,
    generated: list[int],
) -> None:
    steps = exact_list(value, STEPS, "steps")
    expected_keys = {
        "step_index",
        "forward_token_count",
        "input_token_id",
        "input_position_id",
        "generated_token_position_id",
        "input_origin",
        "feedback_from_step",
        "feedback_matches_previous_generated",
        "generated_token_id",
        "generated_matches_logits_top1",
        "final_hidden",
        "logits",
        "top_10",
    }
    for step_index, raw_step in enumerate(steps):
        label = f"steps[{step_index}]"
        step = exact_keys(raw_step, expected_keys, label)
        if integer(step["step_index"], f"{label}.step_index") != step_index:
            fail(f"{label} is out of order")
        expected_forward = 8 if step_index == 0 else 1
        if integer(step["forward_token_count"], f"{label}.forward_token_count") != expected_forward:
            fail(f"{label}.forward_token_count is invalid")
        expected_input = 8 if step_index == 0 else generated[step_index - 1]
        if integer(step["input_token_id"], f"{label}.input_token_id") != expected_input:
            fail(f"{label}.input_token_id does not implement token feedback")
        if integer(step["input_position_id"], f"{label}.input_position_id") != 7 + step_index:
            fail(f"{label}.input_position_id is invalid")
        if integer(
            step["generated_token_position_id"], f"{label}.generated_token_position_id"
        ) != 8 + step_index:
            fail(f"{label}.generated_token_position_id is invalid")
        if step_index == 0:
            if (
                step["input_origin"] != "prompt_last_token"
                or step["feedback_from_step"] is not None
                or step["feedback_matches_previous_generated"] is not None
            ):
                fail("step zero feedback origin is invalid")
        else:
            if (
                step["input_origin"] != "previous_step_generated_token"
                or step["feedback_from_step"] != step_index - 1
                or step["feedback_matches_previous_generated"] is not True
            ):
                fail(f"{label} does not record the previous generated-token feedback")
        if integer(step["generated_token_id"], f"{label}.generated_token_id") != generated[step_index]:
            fail(f"{label}.generated_token_id differs from generated_token_ids")
        validate_tensor(
            root,
            manifests,
            step["final_hidden"],
            f"steps/step-{step_index:02d}-final-hidden.f32",
            HIDDEN_SIZE,
            f"{label}.final_hidden",
        )
        logits_path = validate_tensor(
            root,
            manifests,
            step["logits"],
            f"steps/step-{step_index:02d}-logits.f32",
            VOCAB_SIZE,
            f"{label}.logits",
        )
        top_10 = recompute_top_10(logits_path)
        validate_top_10(step["top_10"], top_10, f"{label}.top_10")
        matches = generated[step_index] == top_10[0]["token_id"]
        if step["generated_matches_logits_top1"] is not matches or not matches:
            fail(f"{label} generated token does not match logits top-1")


def validate_feedback(value: Any) -> None:
    expected = {
        "feedback_edge_count": 7,
        "all_feedback_edges_match": True,
        "step_zero_uses_prompt_last_token": True,
        "subsequent_steps_use_previous_generated_token": True,
    }
    if value != expected:
        fail("feedback summary differs from the independently checked chain")


def validate(root: Path, contract_only: bool = False) -> dict[str, Any]:
    if not root.is_dir():
        fail(f"oracle directory does not exist: {root}")
    metadata_path = root / "metadata.json"
    try:
        metadata_sha256 = sha256_file(metadata_path)
    except OSError as error:
        fail(f"failed to hash metadata.json: {error}")
    trusted = not contract_only
    if trusted and metadata_sha256 != TRUSTED_METADATA_SHA256:
        fail(
            "metadata SHA-256 does not match the promotion trust anchor; "
            "use --contract-only only for an untrusted self-consistency rerun"
        )
    metadata = load_metadata(metadata_path)
    exact_keys(
        metadata,
        {
            "schema_version",
            "created_utc",
            "source",
            "prompt",
            "generation",
            "execution",
            "environment",
            "generated_token_ids",
            "feedback",
            "steps",
            "artifact_files_excluding_metadata",
        },
        "metadata",
    )
    if metadata["schema_version"] != SCHEMA_VERSION:
        fail("schema_version is invalid")
    validate_created_utc(metadata["created_utc"])
    validate_source(metadata["source"])
    validate_prompt(metadata["prompt"])
    generated = exact_list(metadata["generated_token_ids"], STEPS, "generated_token_ids")
    for index, token_id in enumerate(generated):
        if integer(token_id, f"generated_token_ids[{index}]") < 0 or token_id >= VOCAB_SIZE:
            fail(f"generated_token_ids[{index}] is outside the vocabulary")
    if generated != GENERATED_TOKEN_IDS:
        fail("generated_token_ids differs from the frozen source oracle")
    validate_generation(metadata["generation"], generated)
    validate_execution(metadata["execution"])
    validate_environment(metadata["environment"], trusted)
    manifests = validate_artifact_manifest(root, metadata["artifact_files_excluding_metadata"])
    validate_steps(root, manifests, metadata["steps"], generated)
    validate_feedback(metadata["feedback"])
    return {
        "trusted": trusted,
        "mode": "promotion" if trusted else "contract-only",
        "metadata_sha256": metadata_sha256,
        "generated": generated,
        "artifacts": len(manifests),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract-only",
        action="store_true",
        help="validate an untrusted self-consistent rerun without the fixed metadata SHA-256",
    )
    parser.add_argument("oracle_dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = validate(
        args.oracle_dir.expanduser().resolve(), contract_only=args.contract_only
    )
    print(
        "passed=true "
        f"mode={result['mode']} trusted={str(result['trusted']).lower()} "
        f"steps={STEPS} feedback_edges=7 artifacts={result['artifacts']} "
        f"generated_token_ids={','.join(str(token) for token in result['generated'])} "
        f"metadata_sha256={result['metadata_sha256']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
