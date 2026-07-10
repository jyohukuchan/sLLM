#!/usr/bin/env python3
"""Export real fixed-environment vLLM references for SQ8 serving prompts."""

from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import errno
import functools
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "ullm.sq8.serving_oracle.v1"
PAYLOAD_MANIFEST_SCHEMA_VERSION = "ullm.sq8.serving_oracle_payload_manifest.v1"
INPUT_MANIFEST_SHA256 = (
    "c5b502fe54a5f1563eaf48b8308d7f1d479d11afcbf4cb4a7567bb31b65b61af"
)
DEFAULT_FIXTURE = Path(__file__).resolve().parents[1] / "tests/fixtures/sq8-serving-v0.1"
DEFAULT_MODEL = Path(
    "/home/homelab1/datapool/ai_models/safetensors/Qwen/Qwen3-14B-FP8"
)
DEFAULT_OUTPUT = Path("/tmp/ullm-sq8-serving-vllm-oracles-v0.1")
DEFAULT_PYTHON = Path(
    "/home/homelab1/coding-local/ultimateLLM/"
    "uLLM-project/build/envs/vllm-rocm-nightly/bin/python"
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

SOURCE_IDENTITY = {
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

TOKENIZER_IDENTITY = {
    "tokenizer_class": "Qwen2Tokenizer",
    "revision": SOURCE_IDENTITY["revision"],
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

VLLM_IDENTITY = {
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

PRODUCT_CONTRACT = {
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

GENERATION_CASES = [
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

MODEL_INFO = {
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

QUANTIZATION = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "weight_block_size": [128, 128],
}

np: Any = None
torch: Any = None
LLM: Any = None
SamplingParams: Any = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
    )


def ensure_output_available(path: Path) -> None:
    if os.path.lexists(path):
        raise SystemExit(f"refusing to overwrite existing output: {path}")


def rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("renameat2 is required for atomic no-clobber publication")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(f"refusing to overwrite raced output: {destination}")
    raise OSError(error_number, os.strerror(error_number), str(destination))


def require_unchanged(label: str, before: Any, after: Any) -> None:
    if before != after:
        raise RuntimeError(f"{label} changed while the oracle was being exported")


def strict_json(path: Path) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number: {value}")

    value = json.loads(
        path.read_text(encoding="ascii"),
        object_pairs_hook=pairs_hook,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def expected_prompt(prompt_length: int) -> dict[str, Any]:
    digest = hashlib.sha256()
    for token_id in range(1, prompt_length + 1):
        digest.update(struct.pack("<I", token_id))
    feasible = [
        case["case_id"]
        for case in GENERATION_CASES
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


def regular_file(path: Path, label: str) -> None:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise SystemExit(f"{label} must be a regular non-symlink file")


def read_prompt_tokens(path: Path, prompt_length: int) -> list[int]:
    regular_file(path, f"raw prompt {prompt_length}")
    if path.stat().st_size != prompt_length * 4:
        raise SystemExit(f"raw prompt {prompt_length} has an invalid byte length")
    result = []
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for expected in range(1, prompt_length + 1):
            encoded = handle.read(4)
            if len(encoded) != 4:
                raise SystemExit(f"raw prompt {prompt_length} ended early")
            digest.update(encoded)
            token_id = struct.unpack("<I", encoded)[0]
            if token_id != expected or token_id >= VOCAB_SIZE:
                raise SystemExit(f"raw prompt {prompt_length} violates the token rule")
            result.append(token_id)
        if handle.read(1):
            raise SystemExit(f"raw prompt {prompt_length} has trailing bytes")
    if digest.hexdigest() != expected_prompt(prompt_length)["token_ids_u32_le_sha256"]:
        raise SystemExit(f"raw prompt {prompt_length} has an invalid SHA-256")
    return result


def load_input_contract(fixture_dir: Path) -> dict[str, Any]:
    try:
        info = fixture_dir.lstat()
    except OSError as error:
        raise SystemExit(f"fixture directory is unavailable: {error}") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise SystemExit("fixture directory must be a regular non-symlink directory")
    manifest_path = fixture_dir / "manifest.json"
    regular_file(manifest_path, "input manifest")
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != INPUT_MANIFEST_SHA256:
        raise SystemExit("input fixture manifest does not match the fixed SHA-256")
    try:
        manifest = strict_json(manifest_path)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"failed to parse input fixture manifest: {error}") from error
    fixed_fields = {
        "schema_version": "ullm.sq8.serving_fixtures.v1",
        "fixture_set_id": "qwen3-14b-fp8-sq8-serving-v0.1",
        "status": "input_contract_ready_oracles_pending",
        "source_identity": SOURCE_IDENTITY,
        "tokenizer_identity": TOKENIZER_IDENTITY,
        "vllm_identity": VLLM_IDENTITY,
        "product_contract": PRODUCT_CONTRACT,
        "generation_cases": GENERATION_CASES,
        "raw_prompts": [expected_prompt(length) for length in PROMPT_LENGTHS],
    }
    for key, expected in fixed_fields.items():
        if type(manifest.get(key)) is not type(expected) or manifest.get(key) != expected:
            raise SystemExit(f"input fixture manifest field {key} differs")
    expected_placeholders = [
        {
            "oracle_id": f"vllm-raw-p{length:04d}",
            "placeholder_file": f"oracles/raw-p{length:04d}.pending.json",
            "prompt_id": f"raw-p{length:04d}",
            "status": "pending_real_vllm_export",
        }
        for length in PROMPT_LENGTHS
    ]
    if manifest.get("oracle_placeholders") != expected_placeholders:
        raise SystemExit("input fixture oracle placeholders differ")
    trust = manifest.get("trust")
    if (
        not isinstance(trust, dict)
        or trust.get("promotion_eligible") is not False
        or trust.get("synthetic_oracle_values_forbidden") is not True
        or trust.get("required_real_oracle_schema_version") != SCHEMA_VERSION
    ):
        raise SystemExit("input fixture trust state differs")
    prompts = []
    for prompt_length in PROMPT_LENGTHS:
        prompt = expected_prompt(prompt_length)
        source = fixture_dir / prompt["token_file"]
        prompts.append({"record": prompt, "token_ids": read_prompt_tokens(source, prompt_length)})
    return {
        "manifest_bytes": manifest_path.read_bytes(),
        "manifest_sha256": manifest_sha256,
        "prompts": prompts,
    }


def build_schedule(input_contract: dict[str, Any]) -> list[dict[str, Any]]:
    schedule = []
    for prompt in input_contract["prompts"]:
        record = prompt["record"]
        feasible = set(record["feasible_generation_case_ids"])
        for case_id, max_new_tokens, ignore_eos in EXPORTED_GENERATION_CASES:
            if case_id in feasible:
                schedule.append(
                    {
                        "run_index": len(schedule),
                        "prompt_id": record["prompt_id"],
                        "prompt_tokens": record["prompt_tokens"],
                        "case_id": case_id,
                        "max_new_tokens": max_new_tokens,
                        "ignore_eos": ignore_eos,
                    }
                )
    return schedule


def checkpoint_revision(model_dir: Path) -> dict[str, Any]:
    metadata_dir = model_dir / ".cache/huggingface/download"
    revisions: dict[str, str] = {}
    if metadata_dir.is_dir():
        for path in sorted(metadata_dir.glob("*.metadata")):
            lines = path.read_text(encoding="utf-8").splitlines()
            if lines:
                revisions[path.name.removesuffix(".metadata")] = lines[0]
    unique = sorted(set(revisions.values()))
    return {
        "revision": unique[0] if len(unique) == 1 else None,
        "revision_consistent": len(unique) == 1,
        "per_file_revisions": revisions,
    }


def expected_source_files() -> list[dict[str, Any]]:
    return SOURCE_IDENTITY["checkpoint_files"] + TOKENIZER_IDENTITY["files"]


def verify_model_contract(model_dir: Path) -> dict[str, Any]:
    if not model_dir.is_dir():
        raise SystemExit(f"model directory does not exist: {model_dir}")
    records = []
    for expected in expected_source_files():
        path = model_dir / expected["file"]
        regular_file(path, f"model file {expected['file']}")
        if path.stat().st_size != expected["bytes"]:
            raise SystemExit(f"model file size mismatch: {expected['file']}")
        digest = sha256_file(path)
        if digest != expected["sha256"]:
            raise SystemExit(f"model file SHA-256 mismatch: {expected['file']}")
        records.append(dict(expected))
    revision = checkpoint_revision(model_dir)
    expected_names = {record["file"] for record in expected_source_files()}
    if (
        revision["revision"] != SOURCE_IDENTITY["revision"]
        or revision["revision_consistent"] is not True
        or set(revision["per_file_revisions"]) != expected_names
        or any(
            value != SOURCE_IDENTITY["revision"]
            for value in revision["per_file_revisions"].values()
        )
    ):
        raise SystemExit("checkpoint revision metadata differs from the fixed contract")
    return {"files": records, "revision_metadata": revision}


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def load_runtime_dependencies() -> None:
    global LLM, SamplingParams, np, torch
    try:
        import numpy as numpy_module
        import torch as torch_module
        from vllm import LLM as llm_class
        from vllm import SamplingParams as sampling_params_class
    except ImportError as error:
        raise SystemExit(f"vLLM oracle dependencies are unavailable: {error}") from error
    np = numpy_module
    torch = torch_module
    LLM = llm_class
    SamplingParams = sampling_params_class


def validate_runtime_environment() -> dict[str, Any]:
    actual_packages = {
        "vllm": package_version("vllm"),
        "torch": package_version("torch"),
        "transformers": package_version("transformers"),
    }
    expected_packages = {
        "vllm": VLLM_IDENTITY["package_version"],
        "torch": VLLM_IDENTITY["torch_version"],
        "transformers": VLLM_IDENTITY["transformers_version"],
    }
    if actual_packages != expected_packages:
        raise SystemExit("runtime package versions differ from the fixed contract")
    actual = {
        "python_version": platform.python_version(),
        "torch_git_version": str(torch.version.git_version),
        "torch_hip_version": str(torch.version.hip),
    }
    expected = {
        "python_version": VLLM_IDENTITY["python_version"],
        "torch_git_version": VLLM_IDENTITY["torch_git_version"],
        "torch_hip_version": VLLM_IDENTITY["torch_hip_version"],
    }
    if actual != expected:
        raise SystemExit("runtime Python/PyTorch identity differs from the fixed contract")
    if torch.cuda.device_count() != 1:
        raise SystemExit("exactly one GPU must be visible")
    props = torch.cuda.get_device_properties(0)
    device = {
        "visible_device_index": 0,
        "name": str(props.name),
        "gfx": str(getattr(props, "gcnArchName", "")),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "total_memory_bytes": int(props.total_memory),
    }
    if device != VLLM_IDENTITY["device"]:
        raise SystemExit("visible GPU differs from the fixed R9700 contract")
    return {"packages": actual_packages, **actual, "device": device}


def git_repository(exporter_path: Path) -> Path:
    return Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=exporter_path.parent,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
    )


def git_head(exporter_path: Path) -> str:
    repo = git_repository(exporter_path)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if COMMIT_RE.fullmatch(commit) is None:
        raise RuntimeError("git did not return a full commit identity")
    return commit


def git_identity(exporter_path: Path) -> dict[str, Any]:
    repo = git_repository(exporter_path)
    commit = git_head(exporter_path)
    status_lines = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    relative = exporter_path.relative_to(repo).as_posix()
    exporter_status = [line for line in status_lines if line[3:] == relative]
    return {
        "git_commit": commit,
        "git_worktree_dirty": bool(status_lines),
        "exporter_git_status": exporter_status,
        "exporter_repo_relative_path": relative,
    }


def install_prefill_capture(model: Any) -> dict[str, Any]:
    model._ullm_serving_capture_armed = False
    model._ullm_serving_capture_row = None
    model._ullm_serving_capture_forward_tokens = None
    model._ullm_serving_capture_expected_tokens = None

    def capture_final_norm(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        if not model._ullm_serving_capture_armed:
            return
        if not isinstance(output, (tuple, list)) or not output:
            raise RuntimeError("final norm hook expected normalized hidden-state output")
        hidden = output[0]
        if not torch.is_tensor(hidden) or hidden.numel() % HIDDEN_SIZE != 0:
            raise RuntimeError("final norm hook received an invalid hidden tensor")
        rows = hidden.reshape(-1, HIDDEN_SIZE)
        model._ullm_serving_capture_row = rows[-1].detach().clone()
        model._ullm_serving_capture_forward_tokens = int(rows.shape[0])
        model._ullm_serving_capture_armed = False

    model._ullm_serving_capture_hook = model.model.norm.register_forward_hook(
        capture_final_norm
    )
    quant_method = model.model.layers[0].self_attn.qkv_proj.quant_method
    rope_parameters = getattr(model.config, "rope_parameters", None) or {}
    rope_theta = getattr(model.config, "rope_theta", None)
    if rope_theta is None:
        rope_theta = rope_parameters.get("rope_theta")
    actual = {
        "model_class": type(model).__name__,
        "decoder_layer_class": type(model.model.layers[0]).__name__,
        "decoder_layer_count": len(model.model.layers),
        "final_norm_class": type(model.model.norm).__name__,
        "lm_head_class": type(model.lm_head).__name__,
        "quant_config_class": type(model.quant_config).__name__,
        "qkv_quant_method_class": type(quant_method).__name__,
        "hidden_size": int(model.config.hidden_size),
        "vocab_size": int(model.config.vocab_size),
        "tie_word_embeddings": bool(model.config.tie_word_embeddings),
        "rms_norm_eps": float(model.config.rms_norm_eps),
        "rope_theta": float(rope_theta),
        "head_dim": int(model.config.head_dim),
        "num_attention_heads": int(model.config.num_attention_heads),
        "num_key_value_heads": int(model.config.num_key_value_heads),
    }
    if actual != MODEL_INFO:
        raise RuntimeError("loaded vLLM model structure differs from the fixed contract")
    return actual


def arm_prefill_capture(model: Any, prompt_tokens: int) -> None:
    if model._ullm_serving_capture_armed or model._ullm_serving_capture_row is not None:
        raise RuntimeError("prefill hook was armed with an unreleased prior capture")
    model._ullm_serving_capture_expected_tokens = prompt_tokens
    model._ullm_serving_capture_forward_tokens = None
    model._ullm_serving_capture_armed = True


def tensor_record(root: Path, relative: Path, tensor: Any, elements: int) -> dict[str, Any]:
    if str(tensor.dtype) != "torch.bfloat16" or tensor.numel() != elements:
        raise RuntimeError(f"unexpected source tensor for {relative}")
    host = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().reshape(-1)
    if not bool(torch.isfinite(host).all().item()):
        raise RuntimeError(f"non-finite tensor values in {relative}")
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    host.numpy().astype("<f4", copy=False).tofile(path)
    record = {
        "file": relative.as_posix(),
        "dtype": "f32_le",
        "source_dtype": "torch.bfloat16",
        "shape": [elements],
        "bytes": elements * 4,
        "sha256": sha256_file(path),
    }
    del host
    return record


def collect_prefill_capture(
    model: Any,
    *,
    work_dir: str,
    prompt_id: str,
    persist_payloads: bool,
) -> dict[str, Any]:
    if model._ullm_serving_capture_armed:
        raise RuntimeError("vLLM did not execute a forward after the hook was armed")
    row = model._ullm_serving_capture_row
    forward_tokens = model._ullm_serving_capture_forward_tokens
    expected_tokens = model._ullm_serving_capture_expected_tokens
    if row is None or forward_tokens != expected_tokens:
        raise RuntimeError(
            f"prefill hook mismatch: expected={expected_tokens} actual={forward_tokens}"
        )
    host = row.detach().to(device="cpu", dtype=torch.float32).contiguous().reshape(-1)
    if host.numel() != HIDDEN_SIZE or not bool(torch.isfinite(host).all().item()):
        raise RuntimeError("captured final hidden row is invalid")
    hidden_bytes = host.numpy().astype("<f4", copy=False).tobytes()
    hidden_sha256 = sha256_bytes(hidden_bytes)
    result: dict[str, Any] = {
        "prefill_forward_token_count": int(forward_tokens),
        "prefill_hidden_f32_sha256": hidden_sha256,
    }
    if persist_payloads:
        root = Path(work_dir)
        prompt_root = Path("prompts") / prompt_id
        hidden_record = tensor_record(
            root, prompt_root / "final-hidden.f32le", row, HIDDEN_SIZE
        )
        if hidden_record["sha256"] != hidden_sha256:
            raise RuntimeError("captured hidden hash changed while writing")
        with torch.inference_mode():
            logits = model.compute_logits(row.reshape(1, HIDDEN_SIZE))
        if logits is None or logits.numel() != VOCAB_SIZE:
            raise RuntimeError("vLLM produced an invalid full-logit tensor")
        logits = logits.reshape(VOCAB_SIZE)
        logits_record = tensor_record(
            root, prompt_root / "prefill-logits.f32le", logits, VOCAB_SIZE
        )
        host_logits = logits.detach().to(device="cpu", dtype=torch.float32).numpy()
        if not bool(np.isfinite(host_logits).all()):
            raise RuntimeError("vLLM produced non-finite full logits")
        token_ids = np.arange(VOCAB_SIZE, dtype=np.int64)
        indices = np.lexsort((token_ids, -host_logits))[:TOP_K]
        top_10 = [
            {"token_id": int(token_id), "logit": float(host_logits[token_id])}
            for token_id in indices
        ]
        result.update(
            {
                "final_hidden": hidden_record,
                "logits": logits_record,
                "top_10": top_10,
            }
        )
        del token_ids, indices, host_logits, logits
    del hidden_bytes, host, row
    model._ullm_serving_capture_row = None
    model._ullm_serving_capture_forward_tokens = None
    model._ullm_serving_capture_expected_tokens = None
    return result


def uninstall_prefill_capture(model: Any) -> None:
    model._ullm_serving_capture_hook.remove()
    del model._ullm_serving_capture_hook
    del model._ullm_serving_capture_armed
    del model._ullm_serving_capture_row
    del model._ullm_serving_capture_forward_tokens
    del model._ullm_serving_capture_expected_tokens


def apply_one(llm: Any, operation: Any) -> Any:
    values = llm.apply_model(operation)
    if not isinstance(values, list) or len(values) != 1:
        raise RuntimeError("vLLM apply_model did not return exactly one worker result")
    return values[0]


def write_u32_tokens(root: Path, relative: Path, token_ids: list[int]) -> dict[str, Any]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("wb") as handle:
        for token_id in token_ids:
            if isinstance(token_id, bool) or not 0 <= token_id < VOCAB_SIZE:
                raise RuntimeError("vLLM returned an invalid token ID")
            encoded = struct.pack("<I", token_id)
            handle.write(encoded)
            digest.update(encoded)
    return {
        "file": relative.as_posix(),
        "dtype": "u32_le",
        "generated_tokens": len(token_ids),
        "bytes": len(token_ids) * 4,
        "sha256": digest.hexdigest(),
    }


def payload_manifest(root: Path) -> dict[str, Any]:
    excluded = {"metadata.json", "payload-manifest.json", "SHA256SUMS"}
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        files.append(
            {
                "file": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {"schema_version": PAYLOAD_MANIFEST_SCHEMA_VERSION, "files": files}


def write_sums(root: Path) -> None:
    paths = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    (root / "SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n"
            for path in paths
        ),
        encoding="ascii",
    )


def main() -> int:
    args = parse_args()
    fixture_dir = Path(os.path.abspath(args.fixture_dir.expanduser()))
    model_dir = args.model_dir.expanduser().resolve()
    output_dir = Path(os.path.abspath(args.output_dir.expanduser()))
    exporter_path = Path(__file__).resolve()
    exporter_bytes = exporter_path.read_bytes()
    exporter_sha256 = sha256_bytes(exporter_bytes)
    ensure_output_available(output_dir)
    if os.environ.get("ROCR_VISIBLE_DEVICES") != "1":
        raise SystemExit("ROCR_VISIBLE_DEVICES must be exactly 1")
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") not in (None, "0"):
        raise SystemExit("VLLM_ENABLE_V1_MULTIPROCESSING must be 0")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    if Path(sys.executable).resolve() != DEFAULT_PYTHON.resolve():
        raise SystemExit(f"oracle must run with the fixed interpreter: {DEFAULT_PYTHON}")

    initial_input = load_input_contract(fixture_dir)
    schedule = build_schedule(initial_input)
    if len(schedule) != 21 or schedule[-1]["prompt_id"] != "raw-p4095":
        raise RuntimeError("fixed prompt/case schedule is invalid")
    initial_model = verify_model_contract(model_dir)
    source_git = git_identity(exporter_path)
    load_runtime_dependencies()
    runtime_environment = validate_runtime_environment()

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.incomplete-", dir=output_dir.parent)
    )
    llm: Any = None
    success = False
    try:
        captured_exporter = work_dir / "captured-exporter.py"
        captured_exporter.write_bytes(exporter_bytes)
        captured_exporter.chmod(captured_exporter.stat().st_mode | stat.S_IXUSR)
        (work_dir / "input-fixture-manifest.json").write_bytes(
            initial_input["manifest_bytes"]
        )
        prompt_by_id = {
            prompt["record"]["prompt_id"]: prompt for prompt in initial_input["prompts"]
        }
        for prompt in initial_input["prompts"]:
            destination = work_dir / "inputs" / f"{prompt['record']['prompt_id']}.u32le"
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("wb") as handle:
                for token_id in prompt["token_ids"]:
                    handle.write(struct.pack("<I", token_id))

        llm = LLM(
            model=str(model_dir),
            tokenizer=str(model_dir),
            dtype="auto",
            quantization="fp8",
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            max_model_len=CONTEXT_LENGTH,
            max_num_seqs=1,
            max_num_batched_tokens=CONTEXT_LENGTH,
            kv_cache_memory_bytes=KV_CACHE_MEMORY_BYTES,
            enforce_eager=True,
            enable_prefix_caching=False,
            async_scheduling=False,
            disable_log_stats=True,
            seed=0,
        )
        model_info = apply_one(llm, install_prefill_capture)
        prompt_results: dict[str, dict[str, Any]] = {}
        run_records = []
        for run in schedule:
            prompt = prompt_by_id[run["prompt_id"]]
            apply_one(
                llm,
                functools.partial(
                    arm_prefill_capture, prompt_tokens=run["prompt_tokens"]
                ),
            )
            sampling = SamplingParams(
                temperature=0.0,
                max_tokens=run["max_new_tokens"],
                min_tokens=(run["max_new_tokens"] if run["ignore_eos"] else 0),
                ignore_eos=run["ignore_eos"],
                stop_token_ids=([] if run["ignore_eos"] else list(EOS_TOKEN_IDS)),
                seed=0,
            )
            outputs = llm.generate(
                [{"prompt_token_ids": prompt["token_ids"]}],
                sampling,
                use_tqdm=False,
            )
            if len(outputs) != 1 or len(outputs[0].outputs) != 1:
                raise RuntimeError("vLLM returned an unexpected request/output count")
            output = outputs[0].outputs[0]
            generated = [int(token_id) for token_id in output.token_ids]
            if not generated or len(generated) > run["max_new_tokens"]:
                raise RuntimeError("vLLM returned an invalid generated-token count")
            persist = run["case_id"] == "greedy-g1"
            captured = apply_one(
                llm,
                functools.partial(
                    collect_prefill_capture,
                    work_dir=str(work_dir),
                    prompt_id=run["prompt_id"],
                    persist_payloads=persist,
                ),
            )
            result = prompt_results.setdefault(
                run["prompt_id"],
                {
                    "prompt": prompt,
                    "prefill": None,
                    "generation_cases": [],
                    "sequence_records": [],
                },
            )
            if persist:
                result["prefill"] = captured
            elif captured["prefill_hidden_f32_sha256"] != result["prefill"][
                "prefill_hidden_f32_sha256"
            ]:
                raise RuntimeError("prefill final hidden changed across generation cases")
            top1 = int(result["prefill"]["top_10"][0]["token_id"])
            if generated[0] != top1:
                raise RuntimeError("greedy generated token does not match prefill logits top-1")
            finish_reason = str(output.finish_reason)
            if finish_reason not in {"length", "stop"}:
                raise RuntimeError(f"unexpected vLLM finish reason: {finish_reason!r}")
            if run["ignore_eos"]:
                if (
                    len(generated) != run["max_new_tokens"]
                    or finish_reason != "length"
                ):
                    raise RuntimeError(
                        "ignore-EOS boundary generation did not reach exact length"
                    )
            else:
                if finish_reason == "length" and len(generated) != run["max_new_tokens"]:
                    raise RuntimeError("length finish did not reach max_new_tokens")
                if finish_reason == "stop" and generated[-1] not in EOS_TOKEN_IDS:
                    raise RuntimeError("stop finish did not end with a fixed EOS token")
            if result["sequence_records"]:
                previous = result["sequence_records"][-1]
                previous_tokens = previous["token_ids"]
                if generated[: len(previous_tokens)] != previous_tokens:
                    raise RuntimeError(
                        "greedy generation cases are not prefix-consistent"
                    )
                if (
                    previous["finish_reason"] == "stop"
                    and not run["ignore_eos"]
                    and generated != previous_tokens
                ):
                    raise RuntimeError(
                        "normal generation continued after a prior EOS stop"
                    )
            token_record = write_u32_tokens(
                work_dir,
                Path("prompts") / run["prompt_id"] / f"{run['case_id']}.u32le",
                generated,
            )
            result["generation_cases"].append(
                {
                    "case_id": run["case_id"],
                    "max_new_tokens": run["max_new_tokens"],
                    "ignore_eos": run["ignore_eos"],
                    "generated_tokens": len(generated),
                    "finish_reason": finish_reason,
                    "token_file": token_record["file"],
                    "token_file_bytes": token_record["bytes"],
                    "token_ids_u32_le_sha256": token_record["sha256"],
                    "first_token_matches_prefill_top1": True,
                }
            )
            result["sequence_records"].append(
                {
                    "case_id": run["case_id"],
                    "finish_reason": finish_reason,
                    "ignore_eos": run["ignore_eos"],
                    "token_ids": generated,
                }
            )
            run_records.append(
                {
                    "run_index": run["run_index"],
                    "prompt_id": run["prompt_id"],
                    "case_id": run["case_id"],
                    "prefill_forward_token_count": captured[
                        "prefill_forward_token_count"
                    ],
                    "captured_final_norm_rows": 1,
                    "prefill_hidden_f32_sha256": captured[
                        "prefill_hidden_f32_sha256"
                    ],
                }
            )

        apply_one(llm, uninstall_prefill_capture)
        require_unchanged(
            "input fixture",
            initial_input,
            load_input_contract(fixture_dir),
        )
        require_unchanged(
            "model files and revision metadata",
            initial_model,
            verify_model_contract(model_dir),
        )
        require_unchanged(
            "exporter source",
            exporter_sha256,
            sha256_file(exporter_path),
        )
        require_unchanged(
            "exporter source commit",
            source_git["git_commit"],
            git_head(exporter_path),
        )
        prompts_metadata = []
        for prompt_length in PROMPT_LENGTHS:
            prompt_id = f"raw-p{prompt_length:04d}"
            result = prompt_results[prompt_id]
            prefill = result["prefill"]
            input_relative = f"inputs/{prompt_id}.u32le"
            prompts_metadata.append(
                {
                    "prompt_id": prompt_id,
                    "prompt_tokens": prompt_length,
                    "input": {
                        "file": input_relative,
                        "dtype": "u32_le",
                        "bytes": prompt_length * 4,
                        "sha256": result["prompt"]["record"][
                            "token_ids_u32_le_sha256"
                        ],
                        "token_rule": PRODUCT_CONTRACT["prompt_rule"],
                        "position_start": 0,
                        "position_end_inclusive": prompt_length - 1,
                        "attention": "causal",
                    },
                    "prefill": {
                        "capture_case_id": "greedy-g1",
                        "forward_token_count": prefill[
                            "prefill_forward_token_count"
                        ],
                        "final_hidden": prefill["final_hidden"],
                        "logits": prefill["logits"],
                        "top_10": prefill["top_10"],
                    },
                    "generation_cases": result["generation_cases"],
                }
            )

        payload_value = payload_manifest(work_dir)
        payload_path = work_dir / "payload-manifest.json"
        write_json(payload_path, payload_value)
        payload_record = {
            "file": "payload-manifest.json",
            "bytes": payload_path.stat().st_size,
            "sha256": sha256_file(payload_path),
        }
        exporter_record = {
            **source_git,
            "captured_file": "captured-exporter.py",
            "captured_file_bytes": len(exporter_bytes),
            "captured_file_sha256": exporter_sha256,
        }
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "status": "captured_real_vllm",
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "source_fixture": {
                "fixture_set_id": "qwen3-14b-fp8-sq8-serving-v0.1",
                "manifest_file": "input-fixture-manifest.json",
                "manifest_sha256": initial_input["manifest_sha256"],
            },
            "source_model": {
                "identity": SOURCE_IDENTITY,
                "tokenizer_identity": TOKENIZER_IDENTITY,
                "revision_metadata": initial_model["revision_metadata"],
            },
            "execution": {
                "identity": VLLM_IDENTITY,
                "environment": runtime_environment,
                "quantization": QUANTIZATION,
                "model_info": model_info,
                "engine": {
                    "max_model_len": CONTEXT_LENGTH,
                    "max_num_batched_tokens": CONTEXT_LENGTH,
                    "kv_cache_memory_bytes": KV_CACHE_MEMORY_BYTES,
                    "v1_multiprocessing": False,
                    "seed": 0,
                },
                "sampling": {
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
            },
            "capture": {
                "one_model_load": True,
                "runs_sequential": True,
                "maximum_concurrent_requests": 1,
                "run_order": "prompt_length_then_generation_length_ascending",
                "run_count": len(run_records),
                "hook_semantics": "first_forward_final_norm_last_row_only",
                "captured_final_norm_rows_per_run": 1,
                "full_logits_resident_limit": 1,
                "full_logits_capture_case_id_per_prompt": "greedy-g1",
                "runs": run_records,
            },
            "prompts": prompts_metadata,
            "exporter": exporter_record,
            "payload_manifest": payload_record,
        }
        metadata_path = work_dir / "metadata.json"
        write_json(metadata_path, metadata)
        write_sums(work_dir)
        metadata_sha256 = sha256_file(metadata_path)
        rename_noreplace(work_dir, output_dir)
        success = True
        print(
            json.dumps(
                {
                    "exported": True,
                    "output_dir": str(output_dir),
                    "prompt_count": len(prompts_metadata),
                    "run_count": len(run_records),
                    "g512_executed": True,
                    "metadata_sha256": metadata_sha256,
                    "payload_manifest_sha256": payload_record["sha256"],
                },
                sort_keys=True,
            )
        )
    finally:
        if llm is not None:
            engine = getattr(llm, "llm_engine", None)
            if engine is not None and hasattr(engine, "shutdown"):
                try:
                    engine.shutdown()
                except Exception as error:
                    print(f"shutdown warning: {error!r}", file=sys.stderr)
        try:
            from vllm.distributed.parallel_state import (
                destroy_distributed_environment,
                destroy_model_parallel,
            )

            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception as error:
            print(f"distributed cleanup warning: {error!r}", file=sys.stderr)
        del llm
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if not success and work_dir.exists():
            shutil.rmtree(work_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
