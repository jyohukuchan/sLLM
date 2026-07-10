#!/usr/bin/env python3
"""Export deterministic SQ8 serving input contracts and pending oracle manifests."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import shutil
import stat
import struct
import tempfile
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "ullm.sq8.serving_fixtures.v1"
ORACLE_PLACEHOLDER_SCHEMA_VERSION = "ullm.sq8.serving_oracle_placeholder.v1"
REAL_ORACLE_SCHEMA_VERSION = "ullm.sq8.serving_oracle.v1"
OPENWEBUI_CAPTURE_SCHEMA_VERSION = "ullm.openwebui.interop_capture.v1"
DEFAULT_OUTPUT = Path("tests/fixtures/sq8-serving-v0.1")
PROMPT_LENGTHS = (1, 8, 32, 128, 512, 4095)
CHAT_PROMPT_LENGTHS = (32, 128, 512, 2048, 3584)
CHAT_TEMPLATE_MANIFEST_SHA256 = (
    "6324b74e2604b86d46bf2dfdc259c1ca68d8cc9a47e90bfb765919f4aa9d54e0"
)
VOCAB_SIZE = 151_936
HIDDEN_SIZE = 5_120
CONTEXT_LENGTH = 4_096
EOS_TOKEN_IDS = (151_645, 151_643)

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

COMPARISON_CONTRACT = {
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

GENERATION_CASES = [
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

OPENWEBUI_CAPTURE_IDENTITY = {
    "product": "OpenWebUI",
    "version": "v0.9.4",
    "source_revision": "f51d2b026f1b0e7283b15f093412be8b67d24770",
    "image_digest": (
        "sha256:a6da0c292081d810a396ce786a10536d0b1b9ba2925dcca20ebb03f9fa90dbff"
    ),
}

OPENWEBUI_STREAM_REQUEST = {
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

OPENWEBUI_NONSTREAM_REQUEST = {
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

OPENWEBUI_CAPTURE = {
    "schema_version": OPENWEBUI_CAPTURE_SCHEMA_VERSION,
    "status": "captured_sanitized",
    "evidence_scope": "forwarded_request_bodies_only",
    "endpoint": "/api/chat/completions",
    "identity": OPENWEBUI_CAPTURE_IDENTITY,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--chat-template-dir",
        type=Path,
        default=DEFAULT_OUTPUT / "chat-template",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def write_prompt(path: Path, prompt_length: int) -> str:
    digest = hashlib.sha256()
    with path.open("wb") as handle:
        for token_id in range(1, prompt_length + 1):
            encoded = struct.pack("<I", token_id)
            handle.write(encoded)
            digest.update(encoded)
    return digest.hexdigest()


def feasible_generation_cases(prompt_length: int) -> list[str]:
    return [
        case["case_id"]
        for case in GENERATION_CASES
        if prompt_length + case["max_new_tokens"] <= CONTEXT_LENGTH
    ]


def oracle_placeholder(prompt: dict[str, Any]) -> dict[str, Any]:
    generation_outputs = [
        {
            "case_id": case_id,
            "token_file": None,
            "token_file_bytes": None,
            "token_ids_u32_le_sha256": None,
            "generated_tokens": None,
            "status": "pending",
        }
        for case_id in prompt["feasible_generation_case_ids"]
    ]
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
            "greedy_generation": generation_outputs,
        },
        "trust": {
            "synthetic_oracle_values_forbidden": True,
            "real_export_required": True,
            "metadata_sha256_anchor": None,
            "payload_manifest_sha256_anchor": None,
            "real_exporter_source_commit": None,
        },
    }


def copy_chat_template_fixture(source: Path, destination: Path) -> None:
    source = source.expanduser().resolve()
    if not source.is_dir():
        raise SystemExit(f"chat-template fixture directory does not exist: {source}")
    manifest = source / "manifest.json"
    if not manifest.is_file() or manifest.is_symlink():
        raise SystemExit("chat-template manifest must be a regular non-symlink file")
    if sha256_file(manifest) != CHAT_TEMPLATE_MANIFEST_SHA256:
        raise SystemExit("chat-template manifest differs from the frozen trusted export")
    for path in source.rglob("*"):
        if path.is_symlink():
            raise SystemExit(f"chat-template fixture contains a symlink: {path}")
        if not path.is_file() and not path.is_dir():
            raise SystemExit(f"chat-template fixture contains an unsupported entry: {path}")
    shutil.copytree(source, destination)


def artifact_records(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in {"manifest.json", "SHA256SUMS"}:
            continue
        kind = (
            "raw_prompt"
            if relative.startswith("raw/")
            else "oracle_placeholder"
            if relative.startswith("oracles/")
            else "openwebui_capture_metadata"
            if relative == "openwebui/capture.json"
            else "openwebui_forwarded_request"
            if relative.startswith("openwebui/")
            else "chat_template_fixture"
            if relative.startswith("chat-template/")
            else "contract_fixture"
        )
        records.append(
            {
                "file": relative,
                "kind": kind,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def export_fixture_set(root: Path, chat_template_dir: Path) -> None:
    (root / "raw").mkdir(parents=True)
    (root / "oracles").mkdir()
    (root / "openwebui").mkdir()
    prompts = []
    oracle_records = []
    for prompt_length in PROMPT_LENGTHS:
        prompt_id = f"raw-p{prompt_length:04d}"
        relative = f"raw/prompt-{prompt_length:04d}.u32le"
        path = root / relative
        digest = write_prompt(path, prompt_length)
        prompt = {
            "prompt_id": prompt_id,
            "prompt_tokens": prompt_length,
            "position_start": 0,
            "position_end_inclusive": prompt_length - 1,
            "first_token_id": 1,
            "last_token_id": prompt_length,
            "token_file": relative,
            "token_file_bytes": prompt_length * 4,
            "token_ids_u32_le_sha256": digest,
            "feasible_generation_case_ids": feasible_generation_cases(prompt_length),
        }
        prompts.append(prompt)
        placeholder_relative = f"oracles/{prompt_id}.pending.json"
        write_json(root / placeholder_relative, oracle_placeholder(prompt))
        oracle_records.append(
            {
                "oracle_id": f"vllm-{prompt_id}",
                "prompt_id": prompt_id,
                "status": "pending_real_vllm_export",
                "placeholder_file": placeholder_relative,
            }
        )

    copy_chat_template_fixture(chat_template_dir, root / "chat-template")
    write_json(root / "openwebui/stream-request.json", OPENWEBUI_STREAM_REQUEST)
    write_json(root / "openwebui/nonstream-request.json", OPENWEBUI_NONSTREAM_REQUEST)
    write_json(root / "openwebui/capture.json", OPENWEBUI_CAPTURE)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "fixture_set_id": "qwen3-14b-fp8-sq8-serving-v0.1",
        "status": "input_contract_ready_oracles_pending",
        "source_identity": SOURCE_IDENTITY,
        "tokenizer_identity": TOKENIZER_IDENTITY,
        "vllm_identity": VLLM_IDENTITY,
        "product_contract": PRODUCT_CONTRACT,
        "comparison_contract": COMPARISON_CONTRACT,
        "generation_cases": GENERATION_CASES,
        "raw_prompts": prompts,
        "oracle_placeholders": oracle_records,
        "chat_template_fixture": {
            "status": "ready_independent_recompute_passed",
            "directory": "chat-template",
            "manifest_file": "chat-template/manifest.json",
            "manifest_sha256": CHAT_TEMPLATE_MANIFEST_SHA256,
            "exact_prompt_lengths": list(CHAT_PROMPT_LENGTHS),
            "validator": "tools/validate-sq8-chat-template-fixtures.py",
        },
        "openwebui_interop_capture": {
            "status": "captured_sanitized",
            "capture_file": "openwebui/capture.json",
            "stream_request_file": "openwebui/stream-request.json",
            "nonstream_request_file": "openwebui/nonstream-request.json",
        },
        "artifact_files_excluding_manifest_and_sums": artifact_records(root),
        "trust": {
            "fixture_kind": "contract_and_pending_oracle_manifest",
            "promotion_eligible": False,
            "synthetic_oracle_values_forbidden": True,
            "required_real_oracle_schema_version": REAL_ORACLE_SCHEMA_VERSION,
            "trusted_manifest_anchor_location": (
                "tools/validate-sq8-serving-fixtures.py:TRUSTED_MANIFEST_SHA256"
            ),
        },
    }
    write_json(root / "manifest.json", manifest)
    sum_paths = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    sums = "".join(
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n"
        for path in sum_paths
    )
    (root / "SHA256SUMS").write_text(sums, encoding="ascii")


def main() -> int:
    args = parse_args()
    # Keep the final path unresolved so lexists() can reject dangling symlinks.
    output = Path(os.path.abspath(args.output_dir.expanduser()))
    ensure_output_available(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        staged = Path(temporary) / output.name
        staged.mkdir()
        export_fixture_set(staged, args.chat_template_dir)
        for path in staged.rglob("*"):
            mode = path.stat().st_mode
            if path.is_file():
                path.chmod(mode | stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        rename_noreplace(staged, output)
    manifest_hash = sha256_file(output / "manifest.json")
    print(
        f"exported=true output={output} prompts={len(PROMPT_LENGTHS)} "
        f"oracles_pending={len(PROMPT_LENGTHS)} manifest_sha256={manifest_hash}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
