#!/usr/bin/env python3
"""Validate deterministic SQ8 serving contracts without promoting pending oracles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import stat
import struct
import sys
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA_VERSION = "ullm.sq8.serving_fixtures.v1"
ORACLE_PLACEHOLDER_SCHEMA_VERSION = "ullm.sq8.serving_oracle_placeholder.v1"
REAL_ORACLE_SCHEMA_VERSION = "ullm.sq8.serving_oracle.v1"
OPENWEBUI_CAPTURE_SCHEMA_VERSION = "ullm.openwebui.interop_capture.v1"
TRUSTED_MANIFEST_SHA256 = "c5b502fe54a5f1563eaf48b8308d7f1d479d11afcbf4cb4a7567bb31b65b61af"
PROMPT_LENGTHS = (1, 8, 32, 128, 512, 4095)
CHAT_PROMPT_LENGTHS = (32, 128, 512, 2048, 3584)
CHAT_TEMPLATE_MANIFEST_SHA256 = (
    "6324b74e2604b86d46bf2dfdc259c1ca68d8cc9a47e90bfb765919f4aa9d54e0"
)
CHAT_TEMPLATE_FILES = {
    "chat-template/manifest.json",
    "chat-template/fixtures/code-block.json",
    "chat-template/fixtures/english-user.json",
    "chat-template/fixtures/exact-p0032.json",
    "chat-template/fixtures/exact-p0128.json",
    "chat-template/fixtures/exact-p0512.json",
    "chat-template/fixtures/exact-p2048.json",
    "chat-template/fixtures/exact-p3584.json",
    "chat-template/fixtures/japanese-user.json",
    "chat-template/fixtures/system-user.json",
    "chat-template/fixtures/two-turn.json",
}
VOCAB_SIZE = 151_936
HIDDEN_SIZE = 5_120
CONTEXT_LENGTH = 4_096
EOS_TOKEN_IDS = (151_645, 151_643)
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

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

EXPECTED_COMPARISON_CONTRACT = {
    "metric_definitions": {
        "relative_l2": "l2(actual-reference)/max(l2(reference),1e-30)",
        "cosine_similarity": "dot(actual,reference)/(l2(actual)*l2(reference))",
        "top_10_overlap": "set_intersection_count_of_token_ids",
    },
    "vllm_source_model_gate": {
        "nonfinite_count": 0,
        "max_relative_l2": 0.20,
        "min_cosine_similarity": 0.98,
        "top_1_exact": True,
        "minimum_top_10_overlap": 3,
    },
    "ullm_path_equivalence_gate": {
        "nonfinite_count": 0,
        "max_relative_l2": 0.10,
        "min_cosine_similarity": 0.995,
        "top_1_exact": True,
        "minimum_top_10_overlap": 5,
    },
    "tensor_contracts": {
        "final_hidden": {"dtype": "f32_le", "shape": [HIDDEN_SIZE]},
        "logits": {"dtype": "f32_le", "shape": [VOCAB_SIZE]},
        "generated_token_ids": {"dtype": "u32_le", "shape": ["generated_tokens"]},
    },
}

EXPECTED_GENERATION_CASES = [
    {
        "case_id": "greedy-g1",
        "max_new_tokens": 1,
        "ignore_eos": False,
        "test_only": False,
    },
    {
        "case_id": "greedy-g8",
        "max_new_tokens": 8,
        "ignore_eos": False,
        "test_only": False,
    },
    {
        "case_id": "greedy-g64",
        "max_new_tokens": 64,
        "ignore_eos": False,
        "test_only": False,
    },
    {
        "case_id": "greedy-g512-ignore-eos-boundary",
        "max_new_tokens": 512,
        "ignore_eos": True,
        "test_only": True,
    },
]

EXPECTED_OPENWEBUI_CAPTURE_IDENTITY = {
    "product": "OpenWebUI",
    "version": "v0.9.4",
    "source_revision": "f51d2b026f1b0e7283b15f093412be8b67d24770",
    "image_digest": (
        "sha256:a6da0c292081d810a396ce786a10536d0b1b9ba2925dcca20ebb03f9fa90dbff"
    ),
}

EXPECTED_OPENWEBUI_STREAM_REQUEST = {
    "model": "ullm-qwen3-14b-sq8",
    "messages": [
        {
            "role": "system",
            "content": "You are the fixed P8-A interoperability fixture.",
        },
        {"role": "user", "content": "Reply with the word fixture."},
    ],
    "stream": True,
    "temperature": 0.6,
    "top_p": 0.95,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "seed": 12_345,
    "max_tokens": 64,
}

EXPECTED_OPENWEBUI_NONSTREAM_REQUEST = {
    "model": "ullm-qwen3-14b-sq8",
    "messages": [
        {
            "role": "system",
            "content": "You are the fixed P8-A interoperability fixture.",
        },
        {"role": "user", "content": "First turn."},
        {"role": "assistant", "content": "First answer."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Reply with the word fixture."}
            ],
        },
    ],
    "stream": False,
    "temperature": 0,
    "max_tokens": 16,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "seed": 12_345,
}

EXPECTED_OPENWEBUI_CAPTURE = {
    "schema_version": OPENWEBUI_CAPTURE_SCHEMA_VERSION,
    "status": "captured_sanitized",
    "evidence_scope": "forwarded_request_bodies_only",
    "endpoint": "/api/chat/completions",
    "identity": EXPECTED_OPENWEBUI_CAPTURE_IDENTITY,
    "request_files": {
        "stream": "openwebui/stream-request.json",
        "nonstream": "openwebui/nonstream-request.json",
    },
    "observed_transformations": {
        "metadata_stripped_before_upstream": True,
        "max_completion_tokens_forwarded_as": "max_tokens",
    },
    "sanitization": {
        "authorization_included": False,
        "cookies_included": False,
        "secrets_included": False,
    },
    "trust": {
        "captured_via_actual_proxy": True,
        "response_payload_captured": False,
        "sq8_numeric_oracle": False,
    },
}


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


def reject_nonfinite_constant(value: str) -> None:
    fail(f"non-finite JSON number is forbidden: {value}")


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        text = raw.decode("ascii")
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
            fail(f"{label} length differs: expected={len(expected)} actual={len(actual)}")
        for index, expected_item in enumerate(expected):
            exact_value(actual[index], expected_item, f"{label}[{index}]")
    elif isinstance(expected, float):
        if not math.isfinite(actual) or actual != expected:
            fail(f"{label} differs from the fixed contract")
    elif actual != expected:
        fail(f"{label} differs from the fixed contract")


def integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        fail(f"{label} must be an integer")
    return value


def sha256_value(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        fail(f"{label} must be a lowercase SHA-256 digest")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_root(root: Path) -> Path:
    try:
        info = root.lstat()
    except OSError as error:
        fail(f"fixture directory is absent: {error}")
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        fail("fixture root must be a regular non-symlink directory")
    return root.resolve(strict=True)


def safe_file(root: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        fail(f"{label} must be a non-empty relative path")
    pure = PurePosixPath(raw)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        fail(f"{label} is unsafe: {raw!r}")
    path = root.joinpath(*pure.parts)
    try:
        info = path.lstat()
    except OSError as error:
        fail(f"missing fixture file {raw}: {error}")
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        fail(f"fixture artifact must be a regular non-symlink file: {raw}")
    try:
        path.resolve(strict=True).relative_to(root)
    except (OSError, ValueError):
        fail(f"fixture artifact escapes root: {raw}")
    return path


def feasible_generation_case_ids(prompt_length: int) -> list[str]:
    return [
        case["case_id"]
        for case in EXPECTED_GENERATION_CASES
        if prompt_length + case["max_new_tokens"] <= CONTEXT_LENGTH
    ]


def expected_prompt(prompt_length: int) -> dict[str, Any]:
    digest = hashlib.sha256()
    for token_id in range(1, prompt_length + 1):
        digest.update(struct.pack("<I", token_id))
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
        "feasible_generation_case_ids": feasible_generation_case_ids(prompt_length),
    }


def expected_oracle_placeholder(prompt: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": ORACLE_PLACEHOLDER_SCHEMA_VERSION,
        "status": "pending_real_vllm_export",
        "oracle_id": f"vllm-{prompt['prompt_id']}",
        "prompt": {
            "prompt_id": prompt["prompt_id"],
            "token_file": prompt["token_file"],
            "prompt_tokens": prompt["prompt_tokens"],
            "token_ids_u32_le_sha256": prompt["token_ids_u32_le_sha256"],
        },
        "requested_outputs": {
            "prefill_final_hidden": {
                "dtype": "f32_le",
                "shape": [HIDDEN_SIZE],
                "file": None,
                "bytes": None,
                "sha256": None,
                "status": "pending",
            },
            "prefill_logits": {
                "dtype": "f32_le",
                "shape": [VOCAB_SIZE],
                "file": None,
                "bytes": None,
                "sha256": None,
                "status": "pending",
            },
            "greedy_generation": [
                {
                    "case_id": case_id,
                    "token_file": None,
                    "token_file_bytes": None,
                    "token_ids_u32_le_sha256": None,
                    "generated_tokens": None,
                    "status": "pending",
                }
                for case_id in prompt["feasible_generation_case_ids"]
            ],
        },
        "trust": {
            "synthetic_oracle_values_forbidden": True,
            "real_export_required": True,
            "metadata_sha256_anchor": None,
            "payload_manifest_sha256_anchor": None,
            "real_exporter_source_commit": None,
        },
    }


def expected_chat_template_record() -> dict[str, Any]:
    return {
        "status": "ready_independent_recompute_passed",
        "directory": "chat-template",
        "manifest_file": "chat-template/manifest.json",
        "manifest_sha256": CHAT_TEMPLATE_MANIFEST_SHA256,
        "exact_prompt_lengths": list(CHAT_PROMPT_LENGTHS),
        "validator": "tools/validate-sq8-chat-template-fixtures.py",
    }


def validate_chat_template_manifest(root: Path) -> None:
    path = safe_file(root, "chat-template/manifest.json", "chat-template manifest")
    if sha256_file(path) != CHAT_TEMPLATE_MANIFEST_SHA256:
        fail("chat-template manifest differs from its independent trust anchor")
    manifest = load_json(path, "chat-template/manifest.json")
    exact_keys(
        manifest,
        {
            "schema_version",
            "fixture_set_id",
            "model",
            "tokenizer",
            "template_options",
            "exact_length_contract",
            "fixture_files",
        },
        "chat_template_manifest",
    )
    exact_value(
        manifest["schema_version"],
        "ullm.sq8.chat_template_fixtures.v1",
        "chat_template_manifest.schema_version",
    )
    exact_value(
        manifest["fixture_set_id"],
        "qwen3-14b-fp8-sq8-chat-template-v0.1",
        "chat_template_manifest.fixture_set_id",
    )
    exact_value(
        manifest["model"].get("id"),
        EXPECTED_SOURCE_IDENTITY["name"],
        "chat_template_manifest.model.id",
    )
    exact_value(
        manifest["model"].get("revision"),
        EXPECTED_SOURCE_IDENTITY["revision"],
        "chat_template_manifest.model.revision",
    )
    exact_value(
        manifest["tokenizer"].get("class"),
        EXPECTED_TOKENIZER_IDENTITY["tokenizer_class"],
        "chat_template_manifest.tokenizer.class",
    )
    exact_value(
        manifest["tokenizer"].get("revision"),
        EXPECTED_TOKENIZER_IDENTITY["revision"],
        "chat_template_manifest.tokenizer.revision",
    )
    exact_value(
        manifest["tokenizer"].get("transformers_version"),
        EXPECTED_VLLM_IDENTITY["transformers_version"],
        "chat_template_manifest.tokenizer.transformers_version",
    )
    exact_value(
        manifest["tokenizer"].get("chat_template"),
        {
            "utf8_bytes": EXPECTED_TOKENIZER_IDENTITY["chat_template_utf8_bytes"],
            "sha256": EXPECTED_TOKENIZER_IDENTITY["chat_template_sha256"],
        },
        "chat_template_manifest.tokenizer.chat_template",
    )
    exact_value(
        manifest["template_options"],
        {"add_generation_prompt": True, "enable_thinking": False},
        "chat_template_manifest.template_options",
    )
    exact_value(
        manifest["exact_length_contract"].get("target_prompt_tokens"),
        list(CHAT_PROMPT_LENGTHS),
        "chat_template_manifest.exact_length_contract.target_prompt_tokens",
    )
    fixture_files = manifest["fixture_files"]
    if not isinstance(fixture_files, list) or len(fixture_files) != 10:
        fail("chat_template_manifest.fixture_files must contain exactly 10 fixtures")


def validate_prompt_payload(path: Path, prompt: dict[str, Any], label: str) -> None:
    expected_bytes = prompt["token_file_bytes"]
    if path.stat().st_size != expected_bytes:
        fail(f"{label} byte length differs from prompt_tokens * 4")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for expected_token in range(1, prompt["prompt_tokens"] + 1):
            encoded = handle.read(4)
            if len(encoded) != 4:
                fail(f"{label} ended before token {expected_token}")
            digest.update(encoded)
            token_id = struct.unpack("<I", encoded)[0]
            if token_id != expected_token or token_id >= VOCAB_SIZE:
                fail(f"{label} token {expected_token - 1} differs from ascending contract")
        if handle.read(1):
            fail(f"{label} has trailing bytes")
    if digest.hexdigest() != prompt["token_ids_u32_le_sha256"]:
        fail(f"{label} SHA-256 does not match payload")


def validate_artifacts(root: Path, manifest: dict[str, Any]) -> set[str]:
    records = manifest["artifact_files_excluding_manifest_and_sums"]
    if not isinstance(records, list):
        fail("artifact file manifest must be a list")
    expected_paths = {
        *(f"raw/prompt-{length:04d}.u32le" for length in PROMPT_LENGTHS),
        *(f"oracles/raw-p{length:04d}.pending.json" for length in PROMPT_LENGTHS),
        *CHAT_TEMPLATE_FILES,
        "openwebui/capture.json",
        "openwebui/nonstream-request.json",
        "openwebui/stream-request.json",
    }
    actual_paths = []
    for index, raw_record in enumerate(records):
        label = f"artifact_files_excluding_manifest_and_sums[{index}]"
        record = exact_keys(raw_record, {"file", "kind", "bytes", "sha256"}, label)
        path = safe_file(root, record["file"], f"{label}.file")
        file_bytes = integer(record["bytes"], f"{label}.bytes")
        if file_bytes < 0 or file_bytes != path.stat().st_size:
            fail(f"{label}.bytes differs from artifact")
        digest = sha256_value(record["sha256"], f"{label}.sha256")
        if digest != sha256_file(path):
            fail(f"{label}.sha256 differs from artifact")
        expected_kind = (
            "raw_prompt"
            if record["file"].startswith("raw/")
            else "oracle_placeholder"
            if record["file"].startswith("oracles/")
            else "openwebui_capture_metadata"
            if record["file"] == "openwebui/capture.json"
            else "openwebui_forwarded_request"
            if record["file"].startswith("openwebui/")
            else "chat_template_fixture"
            if record["file"].startswith("chat-template/")
            else "contract_fixture"
        )
        if record["kind"] != expected_kind:
            fail(f"{label}.kind is invalid")
        actual_paths.append(record["file"])
    if actual_paths != sorted(expected_paths):
        fail("artifact file manifest paths differ from the fixed fixture set")
    return expected_paths


def validate_tree(root: Path, artifact_paths: set[str]) -> None:
    expected_files = artifact_paths | {"manifest.json", "SHA256SUMS"}
    actual_files: set[str] = set()
    expected_directories = {
        "raw",
        "oracles",
        "openwebui",
        "chat-template",
        "chat-template/fixtures",
    }
    actual_directories: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            fail(f"fixture tree contains a symlink: {relative}")
        if stat.S_ISDIR(info.st_mode):
            actual_directories.add(relative)
        elif stat.S_ISREG(info.st_mode):
            actual_files.add(relative)
        else:
            fail(f"fixture tree contains a non-regular entry: {relative}")
    if actual_directories != expected_directories:
        fail("fixture directory set differs from the frozen serving fixture tree")
    if actual_files != expected_files:
        fail(
            f"fixture file set differs: missing={sorted(expected_files - actual_files)} "
            f"extra={sorted(actual_files - expected_files)}"
        )


def validate_sums(root: Path, expected_files: set[str]) -> None:
    sums_path = safe_file(root, "SHA256SUMS", "SHA256SUMS")
    try:
        lines = sums_path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as error:
        fail(f"failed to read SHA256SUMS: {error}")
    expected_sum_files = sorted(expected_files | {"manifest.json"})
    if len(lines) != len(expected_sum_files):
        fail("SHA256SUMS entry count differs")
    parsed_files = []
    for index, line in enumerate(lines):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\s]+)", line)
        if match is None:
            fail(f"SHA256SUMS line {index + 1} is invalid")
        digest, relative = match.groups()
        path = safe_file(root, relative, f"SHA256SUMS line {index + 1}")
        if digest != sha256_file(path):
            fail(f"SHA256SUMS digest mismatch for {relative}")
        parsed_files.append(relative)
    if parsed_files != expected_sum_files:
        fail("SHA256SUMS paths are not the exact sorted fixture file set")


def validate(root: Path, contract_only: bool = False) -> dict[str, Any]:
    root = validate_root(root)
    manifest_path = safe_file(root, "manifest.json", "manifest")
    manifest_sha256 = sha256_file(manifest_path)
    trusted = not contract_only
    if trusted and manifest_sha256 != TRUSTED_MANIFEST_SHA256:
        fail(
            "manifest SHA-256 does not match the promotion trust anchor; "
            "use --contract-only only for an untrusted structural check"
        )
    manifest = load_json(manifest_path, "manifest.json")
    exact_keys(
        manifest,
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
        "manifest",
    )
    exact_value(manifest["schema_version"], SCHEMA_VERSION, "schema_version")
    exact_value(
        manifest["fixture_set_id"],
        "qwen3-14b-fp8-sq8-serving-v0.1",
        "fixture_set_id",
    )
    exact_value(
        manifest["status"],
        "input_contract_ready_oracles_pending",
        "status",
    )
    exact_value(manifest["source_identity"], EXPECTED_SOURCE_IDENTITY, "source_identity")
    exact_value(
        manifest["tokenizer_identity"],
        EXPECTED_TOKENIZER_IDENTITY,
        "tokenizer_identity",
    )
    exact_value(manifest["vllm_identity"], EXPECTED_VLLM_IDENTITY, "vllm_identity")
    exact_value(
        manifest["product_contract"],
        EXPECTED_PRODUCT_CONTRACT,
        "product_contract",
    )
    exact_value(
        manifest["comparison_contract"],
        EXPECTED_COMPARISON_CONTRACT,
        "comparison_contract",
    )
    exact_value(
        manifest["generation_cases"],
        EXPECTED_GENERATION_CASES,
        "generation_cases",
    )
    expected_prompts = [expected_prompt(length) for length in PROMPT_LENGTHS]
    exact_value(manifest["raw_prompts"], expected_prompts, "raw_prompts")
    expected_oracle_records = [
        {
            "oracle_id": f"vllm-{prompt['prompt_id']}",
            "prompt_id": prompt["prompt_id"],
            "status": "pending_real_vllm_export",
            "placeholder_file": f"oracles/{prompt['prompt_id']}.pending.json",
        }
        for prompt in expected_prompts
    ]
    exact_value(
        manifest["oracle_placeholders"],
        expected_oracle_records,
        "oracle_placeholders",
    )
    exact_value(
        manifest["chat_template_fixture"],
        expected_chat_template_record(),
        "chat_template_fixture",
    )
    exact_value(
        manifest["openwebui_interop_capture"],
        {
            "status": "captured_sanitized",
            "capture_file": "openwebui/capture.json",
            "stream_request_file": "openwebui/stream-request.json",
            "nonstream_request_file": "openwebui/nonstream-request.json",
        },
        "openwebui_interop_capture",
    )
    exact_value(
        manifest["trust"],
        {
            "fixture_kind": "contract_and_pending_oracle_manifest",
            "promotion_eligible": False,
            "synthetic_oracle_values_forbidden": True,
            "required_real_oracle_schema_version": REAL_ORACLE_SCHEMA_VERSION,
            "trusted_manifest_anchor_location": (
                "tools/validate-sq8-serving-fixtures.py:TRUSTED_MANIFEST_SHA256"
            ),
        },
        "trust",
    )

    artifact_paths = validate_artifacts(root, manifest)
    validate_tree(root, artifact_paths)
    for prompt, oracle_record in zip(
        expected_prompts, expected_oracle_records, strict=True
    ):
        prompt_path = safe_file(root, prompt["token_file"], prompt["prompt_id"])
        validate_prompt_payload(prompt_path, prompt, prompt["prompt_id"])
        placeholder_path = safe_file(
            root, oracle_record["placeholder_file"], oracle_record["oracle_id"]
        )
        placeholder = load_json(placeholder_path, oracle_record["placeholder_file"])
        exact_value(
            placeholder,
            expected_oracle_placeholder(prompt),
            oracle_record["oracle_id"],
        )
    validate_chat_template_manifest(root)
    exact_value(
        load_json(
            safe_file(root, "openwebui/capture.json", "OpenWebUI capture"),
            "openwebui/capture.json",
        ),
        EXPECTED_OPENWEBUI_CAPTURE,
        "openwebui_capture_payload",
    )
    exact_value(
        load_json(
            safe_file(
                root,
                "openwebui/stream-request.json",
                "OpenWebUI stream request",
            ),
            "openwebui/stream-request.json",
        ),
        EXPECTED_OPENWEBUI_STREAM_REQUEST,
        "openwebui_stream_request",
    )
    exact_value(
        load_json(
            safe_file(
                root,
                "openwebui/nonstream-request.json",
                "OpenWebUI non-stream request",
            ),
            "openwebui/nonstream-request.json",
        ),
        EXPECTED_OPENWEBUI_NONSTREAM_REQUEST,
        "openwebui_nonstream_request",
    )
    validate_sums(root, artifact_paths)
    return {
        "trusted": trusted,
        "mode": "promotion" if trusted else "contract-only",
        "manifest_sha256": manifest_sha256,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract-only",
        action="store_true",
        help="skip the fixed manifest anchor but retain all structural checks",
    )
    parser.add_argument("fixture_dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = validate(
        args.fixture_dir.expanduser(), contract_only=args.contract_only
    )
    print(
        "valid=true oracle_status=pending promotion_eligible=false "
        f"mode={summary['mode']} trusted={str(summary['trusted']).lower()} "
        f"prompts={len(PROMPT_LENGTHS)} manifest_sha256={summary['manifest_sha256']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
