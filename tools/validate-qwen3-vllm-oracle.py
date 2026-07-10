#!/usr/bin/env python3
"""Validate the fixed Qwen3-14B-FP8 M=8 vLLM oracle artifact."""

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


SCHEMA_VERSION = "ullm.qwen3_full_model_oracle.v1"
TOKEN_IDS = list(range(1, 9))
POSITION_IDS = list(range(8))
SEQUENCE_LEN = 8
HIDDEN_SIZE = 5120
VOCAB_SIZE = 151936
LAYER_COUNT = 40
TOP_K = 10
TRUSTED_METADATA_SHA256 = "5caafcd2c976482dd01e51b537593d8924d381a8a9ab076b2082325e22fea39e"
EXPECTED_REVISION = "9a283b4a5efbc09ce247e0ae5b02b744739e525a"
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
EXPECTED_SEMANTICS = {
    "layer_output": (
        "logical post-MLP residual stream, captured as the second output of "
        "the following fused RMSNorm (or final RMSNorm for layer 39)"
    ),
    "final_hidden": "post-final-RMSNorm, immediately before lm_head",
    "logits": (
        "raw lm_head logits before softmax; row i predicts the token after "
        "the prefix ending at input position i"
    ),
    "lm_head_tied": False,
    "lm_head_bias": False,
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
        fail(f"failed to load {path}: {error}")
    if not isinstance(value, dict):
        fail("metadata.json must contain a JSON object")
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


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
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
        fail(f"{label} is not a safe relative path: {raw!r}")
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


def validate_model(metadata: dict[str, Any]) -> None:
    model = exact_keys(
        metadata.get("model"),
        {"name", "local_dir", "revision", "checkpoint_files", "config"},
        "model",
    )
    if model["name"] != "Qwen/Qwen3-14B-FP8":
        fail("model.name is not the fixed source checkpoint")
    if not isinstance(model["local_dir"], str) or not Path(model["local_dir"]).is_absolute():
        fail("model.local_dir must be absolute")
    if model["config"] != EXPECTED_CONFIG:
        fail("model.config differs from the fixed Qwen3-14B-FP8 config")
    if model["checkpoint_files"] != EXPECTED_CHECKPOINT_FILES:
        fail("model.checkpoint_files differs from the fixed checkpoint identity")

    revision = exact_keys(
        model["revision"],
        {"revision", "per_file_revisions", "revision_consistent"},
        "model.revision",
    )
    if revision["revision"] != EXPECTED_REVISION or revision["revision_consistent"] is not True:
        fail("model revision is not the fixed consistent revision")
    per_file = revision["per_file_revisions"]
    if not isinstance(per_file, dict) or set(per_file) != EXPECTED_REVISION_FILES:
        fail("model per-file revision set differs from the fixed checkpoint")
    if any(value != EXPECTED_REVISION for value in per_file.values()):
        fail("model per-file revisions are inconsistent")


def validate_input(metadata: dict[str, Any]) -> None:
    expected = {
        "token_ids": TOKEN_IDS,
        "position_ids": POSITION_IDS,
        "attention": "causal",
        "bos_inserted": False,
        "chat_template_applied": False,
        "eos_ignored_for_single_sampler_cross_check": True,
        "eos_token_id": 151645,
    }
    if metadata.get("input") != expected:
        fail("input differs from the fixed M=8 raw-token contract")


def validate_model_info(value: Any) -> None:
    entries = exact_list(value, 1, "execution.model_info")
    info = exact_keys(
        entries[0],
        {
            "model_class",
            "decoder_layer_class",
            "final_norm_class",
            "lm_head_class",
            "decoder_layer_count",
            "quant_config_class",
            "quant_config_repr",
            "qkv_quant_method_class",
            "config_hidden_size",
            "config_vocab_size",
            "tie_word_embeddings",
            "rms_norm_eps",
            "rope_theta",
            "head_dim",
            "num_attention_heads",
            "num_key_value_heads",
        },
        "execution.model_info[0]",
    )
    expected = {
        "model_class": "Qwen3ForCausalLM",
        "decoder_layer_class": "Qwen3DecoderLayer",
        "final_norm_class": "RMSNorm",
        "lm_head_class": "ParallelLMHead",
        "decoder_layer_count": 40,
        "quant_config_class": "Fp8Config",
        "qkv_quant_method_class": "Fp8LinearMethod",
        "config_hidden_size": 5120,
        "config_vocab_size": 151936,
        "tie_word_embeddings": False,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000.0,
        "head_dim": 128,
        "num_attention_heads": 40,
        "num_key_value_heads": 8,
    }
    for key, expected_value in expected.items():
        if info[key] != expected_value:
            fail(f"execution.model_info[0].{key} differs from the fixed contract")
    quant_repr = info["quant_config_repr"]
    if not isinstance(quant_repr, str) or "quantization.fp8.Fp8Config object" not in quant_repr:
        fail("execution.model_info[0].quant_config_repr is not Fp8Config")


def validate_execution(metadata: dict[str, Any]) -> None:
    execution = exact_keys(
        metadata.get("execution"),
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
            "seed",
            "v1_multiprocessing",
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
        "max_model_len": 9,
        "max_num_seqs": 1,
        "max_num_batched_tokens": 8,
        "kv_cache_memory_bytes": 64 * 1024 * 1024,
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "async_scheduling": False,
        "seed": 0,
        "v1_multiprocessing": False,
    }
    for key, expected_value in expected.items():
        if execution[key] != expected_value:
            fail(f"execution.{key} differs from the fixed contract")
    validate_model_info(execution["model_info"])


def validate_environment(metadata: dict[str, Any]) -> None:
    environment = exact_keys(
        metadata.get("environment"),
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
    for key in ("python", "python_executable", "platform", "torch_git_version", "torch_hip_version"):
        if not isinstance(environment[key], str) or not environment[key]:
            fail(f"environment.{key} must be a non-empty string")
    if not Path(environment["python_executable"]).is_absolute():
        fail("environment.python_executable must be absolute")
    packages = exact_keys(
        environment["packages"],
        {"vllm", "torch", "transformers", "safetensors", "accelerate", "triton", "numpy"},
        "environment.packages",
    )
    if any(not isinstance(value, str) or not value for value in packages.values()):
        fail("all recorded oracle package versions must be non-empty strings")
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
        or not isinstance(gpu["name"], str)
        or not gpu["name"]
    ):
        fail("environment.gpu is not the isolated R9700/gfx1201 contract")


def validate_trusted_promotion_identity(metadata: dict[str, Any]) -> None:
    if metadata["environment"] != TRUSTED_ENVIRONMENT:
        fail("trusted environment differs from the promoted vLLM/ROCm/gfx1201 identity")
    execution = metadata["execution"]
    if execution["backend"] != "vLLM" or execution["dtype"] != "bfloat16":
        fail("trusted execution backend or dtype differs from the promoted identity")


def validate_artifact_manifest(
    root: Path, metadata: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    raw_records = metadata.get("artifact_files_excluding_metadata")
    if not isinstance(raw_records, list):
        fail("artifact_files_excluding_metadata must be a list")
    records: dict[str, dict[str, Any]] = {}
    ordered_names = []
    for index, raw_record in enumerate(raw_records):
        record = exact_keys(raw_record, {"file", "bytes", "sha256"}, f"artifact[{index}]")
        name = record["file"]
        path = safe_artifact_path(root, name, f"artifact[{index}].file")
        if name in records:
            fail(f"duplicate artifact manifest path: {name}")
        ordered_names.append(name)
        actual_bytes = path.stat().st_size
        if integer(record["bytes"], f"artifact[{index}].bytes") != actual_bytes:
            fail(f"artifact byte count mismatch: {name}")
        expected_sha256 = valid_sha256(record["sha256"], f"artifact[{index}].sha256")
        if sha256_file(path) != expected_sha256:
            fail(f"artifact SHA-256 mismatch: {name}")
        records[name] = record
    if ordered_names != sorted(ordered_names):
        fail("artifact manifest must be sorted by path")

    required = {
        "export_oracle.py",
        "rerun-command.sh",
        "final-hidden.f32",
        "logits.f32",
        *(f"layers/layer-{index:02d}-output.f32" for index in range(LAYER_COUNT)),
    }
    allowed = required | {"run.log"}
    if not required.issubset(records) or not set(records).issubset(allowed):
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
            if relative != "layers":
                fail(f"oracle directory contains an unexpected directory: {relative}")
        elif relative != "metadata.json":
            fail(f"oracle directory contains a non-regular entry: {relative}")
    if actual_files != set(records):
        fail("artifact manifest does not exactly cover files excluding metadata.json")
    return records


def compute_tensor_health(path: Path, expected_elements: int) -> dict[str, Any]:
    try:
        import numpy as np
        import torch
    except ImportError as error:
        fail(f"validator requires numpy and torch for health recomputation: {error}")
    values = np.fromfile(path, dtype="<f4")
    if int(values.size) != expected_elements:
        fail(f"decoded element count mismatch for {path.name}")
    tensor = torch.from_numpy(values)
    finite = torch.isfinite(tensor)
    finite_values = tensor[finite]
    result: dict[str, Any] = {
        "elements": int(tensor.numel()),
        "finite_count": int(finite.sum().item()),
        "nan_count": int(torch.isnan(tensor).sum().item()),
        "inf_count": int(torch.isinf(tensor).sum().item()),
    }
    if finite_values.numel():
        result.update(
            {
                "min": float(finite_values.min().item()),
                "max": float(finite_values.max().item()),
                "mean": float(finite_values.mean().item()),
                "std_population": float(finite_values.std(unbiased=False).item()),
                "l2": float(torch.linalg.vector_norm(finite_values).item()),
                "max_abs": float(finite_values.abs().max().item()),
            }
        )
    return result


def validate_health(reported: Any, actual: dict[str, Any], label: str) -> None:
    expected_keys = {
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
    }
    health = exact_keys(reported, expected_keys, f"{label}.health")
    for key in ("elements", "finite_count", "nan_count", "inf_count"):
        if integer(health[key], f"{label}.health.{key}") != actual[key]:
            fail(f"{label}.health.{key} does not match the payload")
    if (
        actual["finite_count"] != actual["elements"]
        or actual["nan_count"] != 0
        or actual["inf_count"] != 0
    ):
        fail(f"{label} contains non-finite values")
    for key in ("min", "max", "mean", "std_population", "l2", "max_abs"):
        value = finite_number(health[key], f"{label}.health.{key}")
        if not math.isclose(value, actual[key], rel_tol=1e-7, abs_tol=1e-7):
            fail(f"{label}.health.{key} does not match the payload")


def validate_tensor_record(
    root: Path,
    manifests: dict[str, dict[str, Any]],
    raw_record: Any,
    label: str,
    expected_file: str,
    expected_shape: list[int],
    expected_semantic: str,
) -> Path:
    record = exact_keys(
        raw_record,
        {"file", "shape", "storage_dtype", "source_dtype", "bytes", "sha256", "health", "semantic"},
        label,
    )
    if (
        record["file"] != expected_file
        or record["shape"] != expected_shape
        or record["storage_dtype"] != "float32_le"
        or record["source_dtype"] != "torch.bfloat16"
        or record["semantic"] != expected_semantic
    ):
        fail(f"{label} identity, shape, dtype, or semantic is invalid")
    elements = math.prod(expected_shape)
    expected_bytes = elements * 4
    if integer(record["bytes"], f"{label}.bytes") != expected_bytes:
        fail(f"{label}.bytes does not match float32 shape")
    digest = valid_sha256(record["sha256"], f"{label}.sha256")
    manifest = manifests.get(expected_file)
    if manifest is None or manifest["bytes"] != expected_bytes or manifest["sha256"] != digest:
        fail(f"{label} does not match the artifact manifest")
    path = safe_artifact_path(root, expected_file, f"{label}.file")
    validate_health(record["health"], compute_tensor_health(path, elements), label)
    return path


def validate_layers(
    root: Path,
    manifests: dict[str, dict[str, Any]],
    value: Any,
) -> None:
    layers = exact_list(value, LAYER_COUNT, "oracle.layers")
    for layer_index, raw_record in enumerate(layers):
        record = exact_keys(
            raw_record,
            {
                "file",
                "shape",
                "storage_dtype",
                "source_dtype",
                "bytes",
                "sha256",
                "health",
                "semantic",
                "layer_index",
            },
            f"oracle.layers[{layer_index}]",
        )
        if record["layer_index"] != layer_index:
            fail(f"oracle.layers[{layer_index}] has the wrong layer_index")
        semantic = (
            "post_mlp_residual_output_materialized_by_final_fused_rms_norm"
            if layer_index == LAYER_COUNT - 1
            else "post_mlp_residual_output_materialized_by_next_fused_rms_norm"
        )
        without_index = dict(record)
        del without_index["layer_index"]
        validate_tensor_record(
            root,
            manifests,
            without_index,
            f"oracle.layers[{layer_index}]",
            f"layers/layer-{layer_index:02d}-output.f32",
            [SEQUENCE_LEN, HIDDEN_SIZE],
            semantic,
        )


def recompute_topk(logits_path: Path) -> list[dict[str, Any]]:
    try:
        import numpy as np
    except ImportError as error:
        fail(f"validator requires numpy for top-k recomputation: {error}")
    logits = np.fromfile(logits_path, dtype="<f4")
    try:
        logits = logits.reshape(SEQUENCE_LEN, VOCAB_SIZE)
    except ValueError as error:
        fail(f"logits payload shape is invalid: {error}")
    token_ids = np.arange(VOCAB_SIZE, dtype=np.int64)
    result = []
    for position, row in enumerate(logits):
        indices = np.lexsort((token_ids, -row))[:TOP_K]
        values = row[indices]
        result.append(
            {
                "position": position,
                "token_ids": [int(value) for value in indices],
                "logits": [float(value) for value in values],
                "top1_top2_margin": float(values[0] - values[1]),
            }
        )
    return result


def validate_topk(reported: Any, actual: list[dict[str, Any]]) -> None:
    entries = exact_list(reported, SEQUENCE_LEN, "oracle.topk_by_position")
    for position, (raw_entry, actual_entry) in enumerate(zip(entries, actual, strict=True)):
        entry = exact_keys(
            raw_entry,
            {"position", "token_ids", "logits", "top1_top2_margin"},
            f"oracle.topk_by_position[{position}]",
        )
        if entry["position"] != position:
            fail(f"oracle.topk_by_position[{position}].position is invalid")
        if entry["token_ids"] != actual_entry["token_ids"]:
            fail(f"oracle.topk_by_position[{position}] token IDs differ from logits")
        reported_logits = exact_list(
            entry["logits"], TOP_K, f"oracle.topk_by_position[{position}].logits"
        )
        for rank, (reported_value, actual_value) in enumerate(
            zip(reported_logits, actual_entry["logits"], strict=True)
        ):
            if finite_number(reported_value, f"topk[{position}].logits[{rank}]") != actual_value:
                fail(f"oracle.topk_by_position[{position}] logits differ from payload")
        if (
            finite_number(entry["top1_top2_margin"], f"topk[{position}].margin")
            != actual_entry["top1_top2_margin"]
        ):
            fail(f"oracle.topk_by_position[{position}] margin differs from logits")


def validate_oracle(
    root: Path, metadata: dict[str, Any], manifests: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    oracle = exact_keys(
        metadata.get("oracle"),
        {"layers", "final_hidden", "logits", "top_k", "topk_tie_breaker", "topk_by_position"},
        "oracle",
    )
    if oracle["top_k"] != TOP_K or oracle["topk_tie_breaker"] != "token_id_ascending":
        fail("oracle top-k contract is invalid")
    validate_layers(root, manifests, oracle["layers"])
    validate_tensor_record(
        root,
        manifests,
        oracle["final_hidden"],
        "oracle.final_hidden",
        "final-hidden.f32",
        [SEQUENCE_LEN, HIDDEN_SIZE],
        "post_final_rms_norm_pre_lm_head",
    )
    logits_path = validate_tensor_record(
        root,
        manifests,
        oracle["logits"],
        "oracle.logits",
        "logits.f32",
        [SEQUENCE_LEN, VOCAB_SIZE],
        "raw_pre_softmax_logits_for_each_prompt_position",
    )
    topk = recompute_topk(logits_path)
    validate_topk(oracle["topk_by_position"], topk)
    return topk


def validate_sampler(metadata: dict[str, Any], topk: list[dict[str, Any]]) -> int:
    sample = exact_keys(
        metadata.get("sampler_cross_check"),
        {"generated_token_ids", "final_position_top1_token_id", "matches"},
        "sampler_cross_check",
    )
    expected = topk[-1]["token_ids"][0]
    if (
        sample["generated_token_ids"] != [expected]
        or sample["final_position_top1_token_id"] != expected
        or sample["matches"] is not True
    ):
        fail("vLLM sampler output does not match final-position logits top-1")
    return expected


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
            "environment",
            "execution",
            "input",
            "model",
            "oracle",
            "sampler_cross_check",
            "semantics",
            "artifact_files_excluding_metadata",
        },
        "metadata",
    )
    if metadata["schema_version"] != SCHEMA_VERSION:
        fail("schema_version is invalid")
    validate_created_utc(metadata["created_utc"])
    validate_model(metadata)
    validate_input(metadata)
    validate_execution(metadata)
    validate_environment(metadata)
    if trusted:
        validate_trusted_promotion_identity(metadata)
    if metadata["semantics"] != EXPECTED_SEMANTICS:
        fail("semantics differs from the fixed oracle contract")
    manifests = validate_artifact_manifest(root, metadata)
    topk = validate_oracle(root, metadata, manifests)
    sampled_token = validate_sampler(metadata, topk)
    return {
        "artifacts": len(manifests),
        "sampled_token": sampled_token,
        "trusted": trusted,
        "mode": "promotion" if trusted else "contract-only",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract-only",
        action="store_true",
        help=(
            "validate an untrusted self-consistent rerun without the fixed metadata "
            "SHA-256 or exact promoted environment"
        ),
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
        f"layers={LAYER_COUNT} positions={SEQUENCE_LEN} top_k={TOP_K} "
        f"sample_token_id={result['sampled_token']} "
        f"artifacts={result['artifacts']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
