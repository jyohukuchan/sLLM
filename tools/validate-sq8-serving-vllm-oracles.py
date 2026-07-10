#!/usr/bin/env python3
"""Independently validate real SQ8 serving vLLM oracle artifacts."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import heapq
import json
import math
import re
import stat
import struct
import sys
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA_VERSION = "ullm.sq8.serving_oracle.v1"
PAYLOAD_MANIFEST_SCHEMA_VERSION = "ullm.sq8.serving_oracle_payload_manifest.v1"
INPUT_MANIFEST_SHA256 = (
    "c5b502fe54a5f1563eaf48b8308d7f1d479d11afcbf4cb4a7567bb31b65b61af"
)
PROMPT_LENGTHS = (1, 8, 32, 128, 512, 4095)
EXPORTED_GENERATION_CASES = (
    ("greedy-g1", 1, False),
    ("greedy-g8", 8, False),
    ("greedy-g64", 64, False),
    ("greedy-g512-ignore-eos-boundary", 512, True),
)
BOUNDARY_GENERATION_CASE_ID = "greedy-g512-ignore-eos-boundary"
CONTEXT_LENGTH = 4096
HIDDEN_SIZE = 5120
VOCAB_SIZE = 151_936
EOS_TOKEN_IDS = (151_645, 151_643)
TOP_K = 10
KV_CACHE_MEMORY_BYTES = 768 * 1024 * 1024
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")

EXPECTED_SOURCE_IDENTITY = {
    "name": "Qwen/Qwen3-14B-FP8",
    "revision": "9a283b4a5efbc09ce247e0ae5b02b744739e525a",
    "artifact_content_sha256": "2243acf1df627ff6ec13840c8ffcf35c77e89205eb36cef7561b85c9c98b9147",
    "package_manifest_sha256": "c2133dfe392f3d5608bde17ed764ae8347c3096c500a58aa235adbeb63d1a0eb",
    "checkpoint_files": [
        {
            "file": "config.json",
            "bytes": 896,
            "sha256": "c5d7d0e8ee42088bd535101d13c71d38c20b5c2afd46ee8fdfba351956233793",
        },
        {
            "file": "model.safetensors.index.json",
            "bytes": 62_044,
            "sha256": "6a9c8e17744118347080916d8f673b881941cf42989ee77266b14dc2062a7151",
        },
        {
            "file": "model-00001-of-00004.safetensors",
            "bytes": 4_922_397_616,
            "sha256": "2c2f93f7639950a7246c54457482696b94aa0e6b1f49d2169f0422f56c1ed370",
        },
        {
            "file": "model-00002-of-00004.safetensors",
            "bytes": 4_955_472_248,
            "sha256": "7831581bc7d03d77707df3ef10b8d90ee1998ee890ea0020b4a62d27079925ba",
        },
        {
            "file": "model-00003-of-00004.safetensors",
            "bytes": 4_892_558_664,
            "sha256": "d57d1788fb339440b12c6917f7f88e18a5cb76e20f0bfacadd9e4e70a49b2a2a",
        },
        {
            "file": "model-00004-of-00004.safetensors",
            "bytes": 1_555_824_768,
            "sha256": "b4bf668aa6f8535dd467a9a3339116b536682b4241972054b783d514cbe84e50",
        },
    ],
}

EXPECTED_TOKENIZER_IDENTITY = {
    "tokenizer_class": "Qwen2Tokenizer",
    "revision": EXPECTED_SOURCE_IDENTITY["revision"],
    "chat_template_utf8_bytes": 4_168,
    "chat_template_sha256": "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8",
    "files": [
        {
            "file": "tokenizer.json",
            "bytes": 11_422_654,
            "sha256": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
        },
        {
            "file": "tokenizer_config.json",
            "bytes": 9_732,
            "sha256": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
        },
        {
            "file": "vocab.json",
            "bytes": 2_776_833,
            "sha256": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
        },
        {
            "file": "merges.txt",
            "bytes": 1_671_853,
            "sha256": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
        },
        {
            "file": "generation_config.json",
            "bytes": 240,
            "sha256": "231c22c0b89ffbbb785d0e68b2f3f922244f263487af79f6542fc82dbee37dbf",
        },
    ],
}

EXPECTED_VLLM_IDENTITY = {
    "backend": "vLLM",
    "runner": "LLM.generate",
    "package_version": "0.23.1rc1.dev618+g8cf7c4d8a.rocm723",
    "source_revision_from_package_version": "8cf7c4d8a",
    "python_version": "3.12.3",
    "torch_version": "2.11.0+gitd0c8b1f",
    "torch_git_version": "d0c8b1f364ecacff4dd8bc06a645d0fb9324cd37",
    "torch_hip_version": "7.2.53211",
    "transformers_version": "5.12.1",
    "dtype": "bfloat16",
    "tensor_parallel_size": 1,
    "pipeline_parallel_size": 1,
    "max_num_seqs": 1,
    "enforce_eager": True,
    "enable_prefix_caching": False,
    "async_scheduling": False,
    "rocr_visible_devices": "1",
    "device": {
        "visible_device_index": 0,
        "name": "AMD Radeon Graphics",
        "gfx": "gfx1201",
        "compute_capability": [12, 0],
        "total_memory_bytes": 34_208_743_424,
    },
}

EXPECTED_PRODUCT_CONTRACT = {
    "context_length": CONTEXT_LENGTH,
    "vocab_size": VOCAB_SIZE,
    "hidden_size": HIDDEN_SIZE,
    "logits_size": VOCAB_SIZE,
    "prompt_lengths": list(PROMPT_LENGTHS),
    "generation_lengths": [1, 8, 64, 512],
    "eos_token_ids": list(EOS_TOKEN_IDS),
    "prompt_rule": "ascending_u32_token_ids_1_through_prompt_length",
    "position_rule": "zero_based_contiguous",
    "attention": "causal",
    "sampling": "greedy_temperature_zero",
}

EXPECTED_GENERATION_CASES = [
    {"case_id": "greedy-g1", "max_new_tokens": 1, "ignore_eos": False, "test_only": False},
    {"case_id": "greedy-g8", "max_new_tokens": 8, "ignore_eos": False, "test_only": False},
    {"case_id": "greedy-g64", "max_new_tokens": 64, "ignore_eos": False, "test_only": False},
    {
        "case_id": BOUNDARY_GENERATION_CASE_ID,
        "max_new_tokens": 512,
        "ignore_eos": True,
        "test_only": True,
    },
]

EXPECTED_MODEL_INFO = {
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

EXPECTED_QUANTIZATION = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "weight_block_size": [128, 128],
}


class ValidationError(ValueError):
    pass


def fail(message: str) -> None:
    raise ValidationError(message)


def exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        fail(
            f"{label} keys differ: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )
    return value


def exact_value(actual: Any, expected: Any, label: str) -> None:
    if type(actual) is not type(expected):
        fail(
            f"{label} type differs: expected={type(expected).__name__} "
            f"actual={type(actual).__name__}"
        )
    if isinstance(expected, dict):
        exact_keys(actual, set(expected), label)
        for key in expected:
            exact_value(actual[key], expected[key], f"{label}.{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            fail(f"{label} length differs")
        for index, item in enumerate(expected):
            exact_value(actual[index], item, f"{label}[{index}]")
    elif isinstance(expected, float):
        if not math.isfinite(actual) or actual != expected:
            fail(f"{label} differs")
    elif actual != expected:
        fail(f"{label} differs")


def integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        fail(f"{label} must be an integer")
    return value


def finite_float(value: Any, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        fail(f"{label} must be a finite float")
    return value


def sha256_value(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        fail(f"{label} must be a lowercase SHA-256")
    return value


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_nonfinite_constant(value: str) -> None:
    fail(f"non-finite JSON number is forbidden: {value}")


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        text = path.read_bytes().decode("ascii")
    except (OSError, UnicodeError) as error:
        fail(f"failed to read {label}: {error}")
    try:
        value = json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite_constant,
        )
    except ValidationError:
        raise
    except json.JSONDecodeError as error:
        fail(f"failed to parse {label}: {error}")
    if not isinstance(value, dict):
        fail(f"{label} must contain an object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_root(raw: Path) -> Path:
    try:
        info = raw.lstat()
    except OSError as error:
        fail(f"oracle directory is unavailable: {error}")
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        fail("oracle root must be a regular non-symlink directory")
    return raw.resolve(strict=True)


def safe_file(root: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        fail(f"{label} must be a non-empty relative path")
    pure = PurePosixPath(raw)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        fail(f"{label} is an unsafe path")
    path = root.joinpath(*pure.parts)
    try:
        info = path.lstat()
    except OSError as error:
        fail(f"missing oracle file {raw}: {error}")
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        fail(f"oracle artifact must be a regular non-symlink file: {raw}")
    try:
        path.resolve(strict=True).relative_to(root)
    except (OSError, ValueError):
        fail(f"oracle artifact escapes root: {raw}")
    return path


def resolve_anchor(
    root: Path, raw_sha256: str | None, raw_file: Path | None
) -> str | None:
    if raw_sha256 is not None:
        return sha256_value(raw_sha256, "--anchor-sha256")
    if raw_file is None:
        return None
    try:
        info = raw_file.lstat()
    except OSError as error:
        fail(f"anchor file is unavailable: {error}")
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        fail("anchor file must be a regular non-symlink file")
    resolved = raw_file.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError:
        pass
    else:
        fail("anchor file must be outside the producer-controlled oracle tree")
    try:
        lines = resolved.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as error:
        fail(f"failed to read anchor file: {error}")
    if len(lines) != 1:
        fail("anchor file must contain exactly one SHA-256 line")
    return sha256_value(lines[0], "anchor file")


def expected_prompt(prompt_length: int) -> dict[str, Any]:
    digest = hashlib.sha256()
    for token_id in range(1, prompt_length + 1):
        digest.update(struct.pack("<I", token_id))
    feasible = [
        case["case_id"]
        for case in EXPECTED_GENERATION_CASES
        if prompt_length + case["max_new_tokens"] <= CONTEXT_LENGTH
    ]
    return {
        "prompt_id": f"raw-p{prompt_length:04d}",
        "prompt_tokens": prompt_length,
        "position_start": 0,
        "position_end_inclusive": prompt_length - 1,
        "first_token_id": 1,
        "last_token_id": prompt_length,
        "token_file": f"raw/prompt-{prompt_length:04d}.u32le",
        "token_file_bytes": prompt_length * 4,
        "token_ids_u32_le_sha256": digest.hexdigest(),
        "feasible_generation_case_ids": feasible,
    }


def expected_exported_cases(prompt_length: int) -> list[tuple[str, int, bool]]:
    return [
        (case_id, max_new_tokens, ignore_eos)
        for case_id, max_new_tokens, ignore_eos in EXPORTED_GENERATION_CASES
        if prompt_length + max_new_tokens <= CONTEXT_LENGTH
    ]


def validate_input_fixture_manifest(value: dict[str, Any]) -> None:
    exact_keys(
        value,
        {
            "schema_version",
            "fixture_set_id",
            "status",
            "source_identity",
            "tokenizer_identity",
            "vllm_identity",
            "product_contract",
            "comparison_contract",
            "generation_cases",
            "raw_prompts",
            "oracle_placeholders",
            "chat_template_fixture",
            "openwebui_interop_capture",
            "artifact_files_excluding_manifest_and_sums",
            "trust",
        },
        "input_fixture_manifest",
    )
    fixed = {
        "schema_version": "ullm.sq8.serving_fixtures.v1",
        "fixture_set_id": "qwen3-14b-fp8-sq8-serving-v0.1",
        "status": "input_contract_ready_oracles_pending",
        "source_identity": EXPECTED_SOURCE_IDENTITY,
        "tokenizer_identity": EXPECTED_TOKENIZER_IDENTITY,
        "vllm_identity": EXPECTED_VLLM_IDENTITY,
        "product_contract": EXPECTED_PRODUCT_CONTRACT,
        "generation_cases": EXPECTED_GENERATION_CASES,
        "raw_prompts": [expected_prompt(length) for length in PROMPT_LENGTHS],
        "oracle_placeholders": [
            {
                "oracle_id": f"vllm-raw-p{length:04d}",
                "placeholder_file": f"oracles/raw-p{length:04d}.pending.json",
                "prompt_id": f"raw-p{length:04d}",
                "status": "pending_real_vllm_export",
            }
            for length in PROMPT_LENGTHS
        ],
    }
    for key, expected in fixed.items():
        exact_value(value[key], expected, f"input_fixture_manifest.{key}")
    trust = value["trust"]
    if (
        not isinstance(trust, dict)
        or trust.get("promotion_eligible") is not False
        or trust.get("synthetic_oracle_values_forbidden") is not True
        or trust.get("required_real_oracle_schema_version") != SCHEMA_VERSION
    ):
        fail("input fixture manifest does not retain the pending trust boundary")


def expected_payload_paths() -> set[str]:
    paths = {"captured-exporter.py", "input-fixture-manifest.json"}
    for prompt_length in PROMPT_LENGTHS:
        prompt_id = f"raw-p{prompt_length:04d}"
        paths.add(f"inputs/{prompt_id}.u32le")
        paths.add(f"prompts/{prompt_id}/final-hidden.f32le")
        paths.add(f"prompts/{prompt_id}/prefill-logits.f32le")
        for case_id, _, _ in expected_exported_cases(prompt_length):
            paths.add(f"prompts/{prompt_id}/{case_id}.u32le")
    return paths


def validate_payload_manifest(root: Path, metadata_record: Any) -> set[str]:
    record = exact_keys(metadata_record, {"file", "bytes", "sha256"}, "payload_manifest")
    exact_value(record["file"], "payload-manifest.json", "payload_manifest.file")
    path = safe_file(root, record["file"], "payload_manifest.file")
    size = integer(record["bytes"], "payload_manifest.bytes")
    if size != path.stat().st_size:
        fail("payload_manifest.bytes differs from the file")
    digest = sha256_value(record["sha256"], "payload_manifest.sha256")
    if digest != sha256_file(path):
        fail("payload_manifest.sha256 differs from the file")
    value = load_json(path, "payload-manifest.json")
    exact_keys(value, {"schema_version", "files"}, "payload_manifest_payload")
    exact_value(
        value["schema_version"],
        PAYLOAD_MANIFEST_SCHEMA_VERSION,
        "payload_manifest_payload.schema_version",
    )
    records = value["files"]
    if not isinstance(records, list):
        fail("payload_manifest_payload.files must be a list")
    expected_paths = expected_payload_paths()
    actual_paths = []
    for index, raw_record in enumerate(records):
        label = f"payload_manifest_payload.files[{index}]"
        artifact = exact_keys(raw_record, {"file", "bytes", "sha256"}, label)
        path = safe_file(root, artifact["file"], f"{label}.file")
        size = integer(artifact["bytes"], f"{label}.bytes")
        if size < 0 or size != path.stat().st_size:
            fail(f"{label}.bytes differs from the artifact")
        digest = sha256_value(artifact["sha256"], f"{label}.sha256")
        if digest != sha256_file(path):
            fail(f"{label}.sha256 differs from the artifact")
        actual_paths.append(artifact["file"])
    if actual_paths != sorted(expected_paths):
        fail("payload manifest paths differ from the fixed oracle tree")
    return expected_paths


def validate_tree(root: Path, payload_paths: set[str]) -> None:
    expected_files = payload_paths | {"metadata.json", "payload-manifest.json", "SHA256SUMS"}
    expected_directories = {"inputs", "prompts"} | {
        f"prompts/raw-p{length:04d}" for length in PROMPT_LENGTHS
    }
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            fail(f"oracle tree contains a symlink: {relative}")
        if stat.S_ISDIR(info.st_mode):
            actual_directories.add(relative)
        elif stat.S_ISREG(info.st_mode):
            actual_files.add(relative)
        else:
            fail(f"oracle tree contains a non-regular entry: {relative}")
    if actual_directories != expected_directories:
        fail("oracle directory set differs from the fixed tree")
    if actual_files != expected_files:
        fail("oracle file set differs from the fixed tree")


def validate_sums(root: Path, payload_paths: set[str]) -> None:
    path = safe_file(root, "SHA256SUMS", "SHA256SUMS")
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as error:
        fail(f"failed to read SHA256SUMS: {error}")
    expected_paths = sorted(payload_paths | {"metadata.json", "payload-manifest.json"})
    if len(lines) != len(expected_paths):
        fail("SHA256SUMS entry count differs")
    actual_paths = []
    for index, line in enumerate(lines):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\s]+)", line)
        if match is None:
            fail(f"SHA256SUMS line {index + 1} is invalid")
        digest, relative = match.groups()
        artifact = safe_file(root, relative, f"SHA256SUMS line {index + 1}")
        if digest != sha256_file(artifact):
            fail(f"SHA256SUMS digest mismatch for {relative}")
        actual_paths.append(relative)
    if actual_paths != expected_paths:
        fail("SHA256SUMS paths differ from the exact sorted file set")


def scan_u32(
    path: Path,
    *,
    expected_count: int,
    ascending: bool,
    label: str,
) -> list[int]:
    if path.stat().st_size != expected_count * 4:
        fail(f"{label} byte count differs from element count")
    values = []
    with path.open("rb") as handle:
        for index in range(expected_count):
            encoded = handle.read(4)
            if len(encoded) != 4:
                fail(f"{label} ended early")
            token_id = struct.unpack("<I", encoded)[0]
            if token_id >= VOCAB_SIZE:
                fail(f"{label} token ID is outside the vocabulary")
            if ascending and token_id != index + 1:
                fail(f"{label} differs from the ascending prompt rule")
            values.append(token_id)
        if handle.read(1):
            fail(f"{label} has trailing bytes")
    return values


def scan_f32(
    path: Path,
    *,
    elements: int,
    top_k: int,
    label: str,
) -> list[dict[str, Any]]:
    if path.stat().st_size != elements * 4:
        fail(f"{label} byte count differs from tensor shape")
    heap: list[tuple[float, int]] = []
    index = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            if len(chunk) % 4:
                fail(f"{label} contains a partial F32 value")
            for (value,) in struct.iter_unpack("<f", chunk):
                if not math.isfinite(value):
                    fail(f"{label} contains a non-finite value at index {index}")
                if top_k:
                    candidate = (float(value), -index)
                    if len(heap) < top_k:
                        heapq.heappush(heap, candidate)
                    elif candidate > heap[0]:
                        heapq.heapreplace(heap, candidate)
                index += 1
    if index != elements:
        fail(f"{label} element count differs")
    return [
        {"token_id": -negative_id, "logit": value}
        for value, negative_id in sorted(heap, reverse=True)
    ]


def validate_tensor_record(
    root: Path,
    raw: Any,
    *,
    relative: str,
    elements: int,
    top_k: int,
    label: str,
) -> list[dict[str, Any]]:
    record = exact_keys(
        raw,
        {"file", "dtype", "source_dtype", "shape", "bytes", "sha256"},
        label,
    )
    expected = {
        "file": relative,
        "dtype": "f32_le",
        "source_dtype": "torch.bfloat16",
        "shape": [elements],
        "bytes": elements * 4,
    }
    for key, value in expected.items():
        exact_value(record[key], value, f"{label}.{key}")
    path = safe_file(root, record["file"], f"{label}.file")
    digest = sha256_value(record["sha256"], f"{label}.sha256")
    if digest != sha256_file(path):
        fail(f"{label}.sha256 differs from the tensor")
    return scan_f32(path, elements=elements, top_k=top_k, label=label)


def validate_top_10(raw: Any, recomputed: list[dict[str, Any]], label: str) -> None:
    if not isinstance(raw, list) or len(raw) != TOP_K:
        fail(f"{label} must contain exactly {TOP_K} entries")
    seen = set()
    for index, (actual, expected) in enumerate(zip(raw, recomputed, strict=True)):
        entry = exact_keys(actual, {"token_id", "logit"}, f"{label}[{index}]")
        token_id = integer(entry["token_id"], f"{label}[{index}].token_id")
        if not 0 <= token_id < VOCAB_SIZE or token_id in seen:
            fail(f"{label}[{index}].token_id is invalid or duplicated")
        seen.add(token_id)
        logit = finite_float(entry["logit"], f"{label}[{index}].logit")
        if token_id != expected["token_id"] or logit != expected["logit"]:
            fail(f"{label} does not match recomputed logits ranking")


def validate_created_utc(value: Any) -> None:
    if not isinstance(value, str):
        fail("created_utc must be a string")
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError as error:
        fail(f"created_utc is invalid: {error}")
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        fail("created_utc must carry a UTC offset")


def validate_source_fixture(root: Path, raw: Any) -> None:
    value = exact_keys(
        raw,
        {"fixture_set_id", "manifest_file", "manifest_sha256"},
        "source_fixture",
    )
    expected = {
        "fixture_set_id": "qwen3-14b-fp8-sq8-serving-v0.1",
        "manifest_file": "input-fixture-manifest.json",
        "manifest_sha256": INPUT_MANIFEST_SHA256,
    }
    exact_value(value, expected, "source_fixture")
    path = safe_file(root, value["manifest_file"], "source_fixture.manifest_file")
    if sha256_file(path) != INPUT_MANIFEST_SHA256:
        fail("copied input fixture manifest SHA-256 differs")
    validate_input_fixture_manifest(load_json(path, "input-fixture-manifest.json"))


def validate_source_model(raw: Any) -> None:
    value = exact_keys(
        raw,
        {"identity", "tokenizer_identity", "revision_metadata"},
        "source_model",
    )
    exact_value(value["identity"], EXPECTED_SOURCE_IDENTITY, "source_model.identity")
    exact_value(
        value["tokenizer_identity"],
        EXPECTED_TOKENIZER_IDENTITY,
        "source_model.tokenizer_identity",
    )
    expected_names = {
        record["file"]
        for record in EXPECTED_SOURCE_IDENTITY["checkpoint_files"]
        + EXPECTED_TOKENIZER_IDENTITY["files"]
    }
    expected_revision = {
        "revision": EXPECTED_SOURCE_IDENTITY["revision"],
        "revision_consistent": True,
        "per_file_revisions": {
            name: EXPECTED_SOURCE_IDENTITY["revision"] for name in expected_names
        },
    }
    exact_value(
        value["revision_metadata"],
        expected_revision,
        "source_model.revision_metadata",
    )


def validate_execution(raw: Any) -> None:
    value = exact_keys(
        raw,
        {"identity", "environment", "quantization", "model_info", "engine", "sampling"},
        "execution",
    )
    exact_value(value["identity"], EXPECTED_VLLM_IDENTITY, "execution.identity")
    expected_environment = {
        "packages": {
            "vllm": EXPECTED_VLLM_IDENTITY["package_version"],
            "torch": EXPECTED_VLLM_IDENTITY["torch_version"],
            "transformers": EXPECTED_VLLM_IDENTITY["transformers_version"],
        },
        "python_version": EXPECTED_VLLM_IDENTITY["python_version"],
        "torch_git_version": EXPECTED_VLLM_IDENTITY["torch_git_version"],
        "torch_hip_version": EXPECTED_VLLM_IDENTITY["torch_hip_version"],
        "device": EXPECTED_VLLM_IDENTITY["device"],
    }
    exact_value(value["environment"], expected_environment, "execution.environment")
    exact_value(value["quantization"], EXPECTED_QUANTIZATION, "execution.quantization")
    exact_value(value["model_info"], EXPECTED_MODEL_INFO, "execution.model_info")
    exact_value(
        value["engine"],
        {
            "max_model_len": CONTEXT_LENGTH,
            "max_num_batched_tokens": CONTEXT_LENGTH,
            "kv_cache_memory_bytes": KV_CACHE_MEMORY_BYTES,
            "v1_multiprocessing": False,
            "seed": 0,
        },
        "execution.engine",
    )
    exact_value(
        value["sampling"],
        {
            "method": "greedy",
            "temperature": 0.0,
            "seed": 0,
            "top_k_recorded": TOP_K,
            "topk_tie_breaker": "logit_descending_token_id_ascending",
            "profiles": {
                "normal_eos_stop": {
                    "case_ids": ["greedy-g1", "greedy-g8", "greedy-g64"],
                    "min_new_tokens": 0,
                    "ignore_eos": False,
                    "stop_token_ids": list(EOS_TOKEN_IDS),
                },
                "ignore_eos_boundary": {
                    "case_ids": [BOUNDARY_GENERATION_CASE_ID],
                    "min_new_tokens": 512,
                    "ignore_eos": True,
                    "stop_token_ids": [],
                },
            },
        },
        "execution.sampling",
    )


def validate_exporter(root: Path, raw: Any) -> None:
    value = exact_keys(
        raw,
        {
            "git_commit",
            "git_worktree_dirty",
            "exporter_git_status",
            "exporter_repo_relative_path",
            "captured_file",
            "captured_file_bytes",
            "captured_file_sha256",
        },
        "exporter",
    )
    if (
        not isinstance(value["git_commit"], str)
        or COMMIT_RE.fullmatch(value["git_commit"]) is None
    ):
        fail("exporter.git_commit must be a full lowercase git commit")
    if type(value["git_worktree_dirty"]) is not bool:
        fail("exporter.git_worktree_dirty must be boolean")
    statuses = value["exporter_git_status"]
    if not isinstance(statuses, list) or len(statuses) > 1:
        fail("exporter.exporter_git_status must contain at most one path status")
    for status in statuses:
        if not isinstance(status, str) or not status.endswith(
            "tools/export-sq8-serving-vllm-oracles.py"
        ):
            fail("exporter.exporter_git_status contains an invalid path")
        if re.fullmatch(
            r"[ MADRCU?!]{2} tools/export-sq8-serving-vllm-oracles\.py",
            status,
        ) is None:
            fail("exporter.exporter_git_status contains an invalid porcelain status")
    if statuses and value["git_worktree_dirty"] is not True:
        fail("exporter git status is inconsistent with git_worktree_dirty=false")
    exact_value(
        value["exporter_repo_relative_path"],
        "tools/export-sq8-serving-vllm-oracles.py",
        "exporter.exporter_repo_relative_path",
    )
    exact_value(value["captured_file"], "captured-exporter.py", "exporter.captured_file")
    path = safe_file(root, value["captured_file"], "exporter.captured_file")
    size = integer(value["captured_file_bytes"], "exporter.captured_file_bytes")
    if size <= 0 or size != path.stat().st_size:
        fail("exporter.captured_file_bytes differs from the captured source")
    digest = sha256_value(value["captured_file_sha256"], "exporter.captured_file_sha256")
    if digest != sha256_file(path):
        fail("exporter.captured_file_sha256 differs from the captured source")


def validate_prompts(root: Path, raw: Any) -> dict[str, str]:
    if not isinstance(raw, list) or len(raw) != len(PROMPT_LENGTHS):
        fail("prompts must contain the six fixed prompt records")
    hidden_hashes: dict[str, str] = {}
    for prompt_index, (value, prompt_length) in enumerate(
        zip(raw, PROMPT_LENGTHS, strict=True)
    ):
        label = f"prompts[{prompt_index}]"
        prompt = exact_keys(
            value,
            {"prompt_id", "prompt_tokens", "input", "prefill", "generation_cases"},
            label,
        )
        prompt_id = f"raw-p{prompt_length:04d}"
        exact_value(prompt["prompt_id"], prompt_id, f"{label}.prompt_id")
        exact_value(prompt["prompt_tokens"], prompt_length, f"{label}.prompt_tokens")
        input_record = exact_keys(
            prompt["input"],
            {
                "file",
                "dtype",
                "bytes",
                "sha256",
                "token_rule",
                "position_start",
                "position_end_inclusive",
                "attention",
            },
            f"{label}.input",
        )
        expected_input = {
            "file": f"inputs/{prompt_id}.u32le",
            "dtype": "u32_le",
            "bytes": prompt_length * 4,
            "sha256": expected_prompt(prompt_length)["token_ids_u32_le_sha256"],
            "token_rule": EXPECTED_PRODUCT_CONTRACT["prompt_rule"],
            "position_start": 0,
            "position_end_inclusive": prompt_length - 1,
            "attention": "causal",
        }
        exact_value(input_record, expected_input, f"{label}.input")
        input_path = safe_file(root, input_record["file"], f"{label}.input.file")
        if sha256_file(input_path) != input_record["sha256"]:
            fail(f"{label}.input SHA-256 differs")
        scan_u32(
            input_path,
            expected_count=prompt_length,
            ascending=True,
            label=f"{label}.input",
        )

        prefill = exact_keys(
            prompt["prefill"],
            {"capture_case_id", "forward_token_count", "final_hidden", "logits", "top_10"},
            f"{label}.prefill",
        )
        exact_value(
            prefill["capture_case_id"], "greedy-g1", f"{label}.prefill.capture_case_id"
        )
        exact_value(
            prefill["forward_token_count"],
            prompt_length,
            f"{label}.prefill.forward_token_count",
        )
        validate_tensor_record(
            root,
            prefill["final_hidden"],
            relative=f"prompts/{prompt_id}/final-hidden.f32le",
            elements=HIDDEN_SIZE,
            top_k=0,
            label=f"{label}.prefill.final_hidden",
        )
        hidden_hash = sha256_value(
            prefill["final_hidden"]["sha256"],
            f"{label}.prefill.final_hidden.sha256",
        )
        hidden_hashes[prompt_id] = hidden_hash
        recomputed_top_10 = validate_tensor_record(
            root,
            prefill["logits"],
            relative=f"prompts/{prompt_id}/prefill-logits.f32le",
            elements=VOCAB_SIZE,
            top_k=TOP_K,
            label=f"{label}.prefill.logits",
        )
        validate_top_10(prefill["top_10"], recomputed_top_10, f"{label}.prefill.top_10")
        top1 = recomputed_top_10[0]["token_id"]

        cases = prompt["generation_cases"]
        expected_cases = expected_exported_cases(prompt_length)
        if not isinstance(cases, list) or len(cases) != len(expected_cases):
            fail(f"{label}.generation_cases length differs")
        sequences: list[tuple[str, str, list[int], bool]] = []
        for case_index, (case, expected_case) in enumerate(
            zip(cases, expected_cases, strict=True)
        ):
            case_label = f"{label}.generation_cases[{case_index}]"
            record = exact_keys(
                case,
                {
                    "case_id",
                    "max_new_tokens",
                    "ignore_eos",
                    "generated_tokens",
                    "finish_reason",
                    "token_file",
                    "token_file_bytes",
                    "token_ids_u32_le_sha256",
                    "first_token_matches_prefill_top1",
                },
                case_label,
            )
            case_id, max_new_tokens, ignore_eos = expected_case
            exact_value(record["case_id"], case_id, f"{case_label}.case_id")
            exact_value(
                record["max_new_tokens"], max_new_tokens, f"{case_label}.max_new_tokens"
            )
            exact_value(record["ignore_eos"], ignore_eos, f"{case_label}.ignore_eos")
            generated_tokens = integer(
                record["generated_tokens"], f"{case_label}.generated_tokens"
            )
            if not 1 <= generated_tokens <= max_new_tokens:
                fail(f"{case_label}.generated_tokens is outside the case limit")
            finish_reason = record["finish_reason"]
            if finish_reason not in {"length", "stop"}:
                fail(f"{case_label}.finish_reason is invalid")
            if ignore_eos:
                if generated_tokens != 512 or finish_reason != "length":
                    fail(f"{case_label} ignore-EOS boundary must emit exactly 512 tokens")
            else:
                if finish_reason == "length" and generated_tokens != max_new_tokens:
                    fail(f"{case_label} length finish did not reach max_new_tokens")
                if generated_tokens < max_new_tokens and finish_reason != "stop":
                    fail(f"{case_label} short generation is not an EOS stop")
            expected_file = f"prompts/{prompt_id}/{case_id}.u32le"
            exact_value(record["token_file"], expected_file, f"{case_label}.token_file")
            exact_value(
                record["token_file_bytes"],
                generated_tokens * 4,
                f"{case_label}.token_file_bytes",
            )
            path = safe_file(root, record["token_file"], f"{case_label}.token_file")
            digest = sha256_value(
                record["token_ids_u32_le_sha256"],
                f"{case_label}.token_ids_u32_le_sha256",
            )
            if digest != sha256_file(path):
                fail(f"{case_label} token SHA-256 differs")
            tokens = scan_u32(
                path,
                expected_count=generated_tokens,
                ascending=False,
                label=case_label,
            )
            if tokens[0] != top1:
                fail(f"{case_label} first token differs from prefill logits top-1")
            exact_value(
                record["first_token_matches_prefill_top1"],
                True,
                f"{case_label}.first_token_matches_prefill_top1",
            )
            if (
                not ignore_eos
                and finish_reason == "stop"
                and tokens[-1] not in EOS_TOKEN_IDS
            ):
                fail(f"{case_label} stop does not end with a fixed EOS token")
            sequences.append((case_id, finish_reason, tokens, ignore_eos))
        for previous, current in zip(sequences, sequences[1:]):
            previous_id, previous_finish, previous_tokens, _previous_ignore_eos = (
                previous
            )
            current_id, _current_finish, current_tokens, current_ignore_eos = current
            if current_tokens[: len(previous_tokens)] != previous_tokens:
                fail(f"{label} greedy sequences are not prefix-consistent")
            if (
                previous_finish == "stop"
                and not current_ignore_eos
                and current_tokens != previous_tokens
            ):
                fail(f"{label} {current_id} continued after {previous_id} stopped on EOS")
    return hidden_hashes


def expected_run_records(hidden_hashes: dict[str, str]) -> list[dict[str, Any]]:
    records = []
    for prompt_length in PROMPT_LENGTHS:
        prompt_id = f"raw-p{prompt_length:04d}"
        for case_id, _, _ in expected_exported_cases(prompt_length):
            records.append(
                {
                    "run_index": len(records),
                    "prompt_id": prompt_id,
                    "case_id": case_id,
                    "prefill_forward_token_count": prompt_length,
                    "captured_final_norm_rows": 1,
                    "prefill_hidden_f32_sha256": hidden_hashes[prompt_id],
                }
            )
    return records


def validate_capture(raw: Any, hidden_hashes: dict[str, str]) -> None:
    value = exact_keys(
        raw,
        {
            "one_model_load",
            "runs_sequential",
            "maximum_concurrent_requests",
            "run_order",
            "run_count",
            "hook_semantics",
            "captured_final_norm_rows_per_run",
            "full_logits_resident_limit",
            "full_logits_capture_case_id_per_prompt",
            "runs",
        },
        "capture",
    )
    expected = {
        "one_model_load": True,
        "runs_sequential": True,
        "maximum_concurrent_requests": 1,
        "run_order": "prompt_length_then_generation_length_ascending",
        "run_count": 21,
        "hook_semantics": "first_forward_final_norm_last_row_only",
        "captured_final_norm_rows_per_run": 1,
        "full_logits_resident_limit": 1,
        "full_logits_capture_case_id_per_prompt": "greedy-g1",
        "runs": expected_run_records(hidden_hashes),
    }
    exact_value(value, expected, "capture")


def validate(
    root_raw: Path,
    anchor_sha256: str | None,
    anchor_file: Path | None,
) -> dict[str, Any]:
    root = validate_root(root_raw)
    anchor = resolve_anchor(root, anchor_sha256, anchor_file)
    metadata_path = safe_file(root, "metadata.json", "metadata.json")
    metadata_sha256 = sha256_file(metadata_path)
    trusted = anchor is not None
    if trusted and anchor != metadata_sha256:
        fail("metadata SHA-256 does not match the supplied promotion trust anchor")
    metadata = load_json(metadata_path, "metadata.json")
    exact_keys(
        metadata,
        {
            "schema_version",
            "status",
            "created_utc",
            "source_fixture",
            "source_model",
            "execution",
            "capture",
            "prompts",
            "exporter",
            "payload_manifest",
        },
        "metadata",
    )
    exact_value(metadata["schema_version"], SCHEMA_VERSION, "metadata.schema_version")
    exact_value(metadata["status"], "captured_real_vllm", "metadata.status")
    validate_created_utc(metadata["created_utc"])
    payload_paths = validate_payload_manifest(root, metadata["payload_manifest"])
    validate_tree(root, payload_paths)
    validate_source_fixture(root, metadata["source_fixture"])
    validate_source_model(metadata["source_model"])
    validate_execution(metadata["execution"])
    validate_exporter(root, metadata["exporter"])
    hidden_hashes = validate_prompts(root, metadata["prompts"])
    validate_capture(metadata["capture"], hidden_hashes)
    validate_sums(root, payload_paths)
    return {
        "trusted": trusted,
        "mode": "promotion" if trusted else "contract-only",
        "metadata_sha256": metadata_sha256,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    anchors = parser.add_mutually_exclusive_group()
    anchors.add_argument("--anchor-sha256")
    anchors.add_argument("--anchor-file", type=Path)
    parser.add_argument("oracle_dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = validate(
        args.oracle_dir.expanduser(),
        anchor_sha256=args.anchor_sha256,
        anchor_file=args.anchor_file.expanduser() if args.anchor_file else None,
    )
    print(
        "valid=true oracle_status=captured_real_vllm "
        f"mode={summary['mode']} trusted={str(summary['trusted']).lower()} "
        f"prompts={len(PROMPT_LENGTHS)} runs=21 "
        f"metadata_sha256={summary['metadata_sha256']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
