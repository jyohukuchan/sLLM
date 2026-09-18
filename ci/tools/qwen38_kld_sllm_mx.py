#!/usr/bin/env python3
"""Run and validate the Qwen3.8 sLLM MXFP6/MXFP8 KLD dump worker.

The Rust worker consumes the verified derived GGUF.  This small launcher keeps
the real Qwen3.8 source and lock visible in the execution record, prepares a
runtime library path for the HIP binary, and validates the common raw-logit
output contract.  It never relabels Qwen3.5 artifacts as Qwen3.8.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
from typing import Any


OUTPUT_SCHEMA = "sllm-qwen38-kld-mx-driver-v2"
INPUT_SCHEMA = "qwen38-kld-manifest-v1"
MODEL_REPO = "Qwen/Qwen3.8-27B"
MODEL_FINGERPRINT = "sha256:0498226db11f8c2446344aa23e185d934050cee7b8b8b90f34587f3283f42276"
VOCAB_SIZE = 248_320
MAX_CASES = 256
MAX_TOKENS = 262_144
EXPECTED_DIMS = {
    "hidden_size": 5120,
    "num_hidden_layers": 64,
    "intermediate_size": 17408,
    "vocab_size": VOCAB_SIZE,
}


class AdapterError(RuntimeError):
    """A source, worker, or output contract error."""


def sha256_bytes(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdapterError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AdapterError(f"JSON root must be an object: {path}")
    return value, raw


def read_input_manifest(path: Path) -> tuple[dict[str, Any], bytes]:
    value, raw = read_json(path)
    if value.get("schema_version") != INPUT_SCHEMA:
        raise AdapterError(f"input manifest schema must be {INPUT_SCHEMA}")
    cases = value.get("cases")
    if not isinstance(cases, list) or not 0 < len(cases) <= MAX_CASES:
        raise AdapterError(f"input manifest cases must be in 1..={MAX_CASES}")
    ids: set[str] = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict) or set(case) - {"id", "token_ids", "positions"}:
            raise AdapterError(f"case {index} has an unknown field")
        case_id = case.get("id")
        if (
            not isinstance(case_id, str)
            or not case_id
            or case_id in ids
            or case_id in {".", ".."}
            or "/" in case_id
            or "\\" in case_id
        ):
            raise AdapterError(f"case {index} id is empty, unsafe, or duplicated")
        ids.add(case_id)
        tokens = case.get("token_ids")
        if (
            not isinstance(tokens, list)
            or not 0 < len(tokens) <= MAX_TOKENS
            or any(
                isinstance(token, bool)
                or not isinstance(token, int)
                or not 0 <= token < VOCAB_SIZE
                for token in tokens
            )
        ):
            raise AdapterError(f"case {case_id} token_ids are invalid")
        positions = case.get("positions")
        if positions is not None and (
            not isinstance(positions, list)
            or not positions
            or len(set(positions)) != len(positions)
            or any(
                isinstance(position, bool)
                or not isinstance(position, int)
                or not 0 <= position < len(tokens)
                for position in positions
            )
        ):
            raise AdapterError(f"case {case_id} positions are invalid")
    return value, raw


def inspect_qwen38_source(source_root: Path) -> dict[str, Any]:
    """Verify the local source is the actual Qwen3.8 BF16 model family."""
    config, config_bytes = read_json(source_root / "config.json")
    index, index_bytes = read_json(source_root / "model.safetensors.index.json")
    text_config = config.get("text_config", config)
    if not isinstance(text_config, dict):
        raise AdapterError("Qwen3.8 source has no text_config object")
    mismatches = {
        name: (text_config.get(name), expected)
        for name, expected in EXPECTED_DIMS.items()
        if text_config.get(name) != expected
    }
    if config.get("architectures") != ["Qwen3_5ForConditionalGeneration"]:
        mismatches["architectures"] = (
            config.get("architectures"),
            ["Qwen3_5ForConditionalGeneration"],
        )
    if config.get("model_type") != "qwen3_5":
        mismatches["model_type"] = (config.get("model_type"), "qwen3_5")
    if mismatches:
        raise AdapterError(f"source is not the expected Qwen3.8-27B layout: {mismatches}")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise AdapterError("Qwen3.8 source index has no weight_map")
    return {
        "source_root": str(source_root),
        "config_sha256": sha256_bytes(config_bytes),
        "index_sha256": sha256_bytes(index_bytes),
        "indexed_tensor_count": len(weight_map),
        "shard_count": len(set(weight_map.values())),
        "architecture": config["architectures"][0],
        "dimensions": EXPECTED_DIMS,
    }


def token_id_hashes(tokens: list[int]) -> set[str]:
    json_bytes = json.dumps(tokens, separators=(",", ":")).encode()
    binary = struct.pack("<" + "i" * len(tokens), *tokens)
    return {hashlib.sha256(json_bytes).hexdigest(), hashlib.sha256(binary).hexdigest()}


def output_contract(output_dir: Path, input_manifest: dict[str, Any], chunk_size: int) -> dict[str, Any]:
    path = output_dir / "manifest.json"
    if not path.is_file() or path.is_symlink():
        raise AdapterError(f"worker did not write output manifest: {path}")
    value, _ = read_json(path)
    if value.get("engine") != "sLLM" or value.get("model") != "Qwen3.8-27B":
        raise AdapterError("worker output is not the Qwen3.8 sLLM manifest")
    if value.get("chunk_size") != chunk_size:
        raise AdapterError("worker output chunk_size differs from the requested chunk size")
    if chunk_size == 1:
        expected_mode = "prefill_1_then_serial_decode_with_last_logits"
        expected_precision = "FP32 last_logits API"
    else:
        expected_mode = "prefill_1_then_decode_block_with_mtp_state_and_logits_resolve"
        expected_precision = "initial FP32 last_logits; subsequent BF16 widened to FP32"
    if value.get("execution_mode") != expected_mode:
        raise AdapterError("worker output execution mode differs from the requested chunk mode")
    if value.get("logit_storage_precision") != expected_precision:
        raise AdapterError("worker output logit storage precision differs from the requested chunk mode")
    if value.get("model_fingerprint") not in (None, MODEL_FINGERPRINT):
        raise AdapterError("worker output model fingerprint differs from Qwen3.8 lock")
    cases = value.get("cases")
    expected = input_manifest["cases"]
    if not isinstance(cases, list) or len(cases) != len(expected):
        raise AdapterError("worker output manifest case count differs from input")
    for item, source in zip(cases, expected):
        case_id = source["id"]
        if not isinstance(item, dict) or item.get("id") != case_id:
            raise AdapterError(f"worker output case differs: {case_id}")
        expected_positions = sorted(source.get("positions") or range(len(source["token_ids"])))
        expected_chunks = [1]
        remaining = max(0, len(source["token_ids"]) - 1)
        while remaining:
            count = min(chunk_size, remaining)
            expected_chunks.append(count)
            remaining -= count
        shape = item.get("shape")
        if shape != [len(expected_positions), VOCAB_SIZE] or item.get("positions") != expected_positions:
            raise AdapterError(f"worker output shape/positions differ: {case_id}")
        if item.get("nonfinite_count") != 0:
            raise AdapterError(f"worker output reports non-finite logits: {case_id}")
        if item.get("actual_chunk_sizes") != expected_chunks:
            raise AdapterError(f"worker actual chunk sizes differ: {case_id}")
        digest = item.get("input_token_ids_sha256", "")
        if not isinstance(digest, str) or digest.removeprefix("sha256:") not in token_id_hashes(source["token_ids"]):
            raise AdapterError(f"worker output token hash differs: {case_id}")
        filename = item.get("logits_file")
        if filename != f"{case_id}.f32" or Path(filename).name != filename:
            raise AdapterError(f"worker output filename is unsafe: {case_id}")
        raw_path = output_dir / filename
        expected_bytes = shape[0] * shape[1] * 4
        if not raw_path.is_file() or raw_path.is_symlink() or raw_path.stat().st_size != expected_bytes:
            raise AdapterError(f"worker raw logits size differs: {case_id}")
    return value


def worker_candidates(repo: Path) -> list[Path]:
    return [
        repo / "target" / "release" / "sllm-qwen38-mx-kld-dump",
        repo / "target" / "debug" / "sllm-qwen38-mx-kld-dump",
    ]


def build_parser() -> argparse.ArgumentParser:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kind", choices=("mxfp6", "mxfp8"), required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--derived-lock", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, default=repo / "docs" / "models" / "locks" / "qwen3.8-27b-bf16.json")
    parser.add_argument("--worker", type=Path, default=None)
    parser.add_argument("--target", choices=("gfx1030", "gfx1201"), required=True)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--kv", choices=("fp16", "kv-mxfp8-e4", "kv-mxfp8-e5"), default="fp16")
    parser.add_argument("--chunk-size", type=int, default=1)
    return parser


def run(args: argparse.Namespace) -> int:
    source_root = args.source_root.resolve()
    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    model_lock = args.model_lock.resolve()
    gguf = args.gguf.resolve()
    derived_lock = args.derived_lock.resolve()
    if not source_root.is_dir() or source_root.is_symlink():
        raise AdapterError(f"source root must be a regular directory: {source_root}")
    if not model_lock.is_file() or model_lock.is_symlink():
        raise AdapterError(f"model lock must be a regular file: {model_lock}")
    if not gguf.is_file() or gguf.is_symlink():
        raise AdapterError(f"GGUF must be a regular file: {gguf}")
    if not derived_lock.is_file() or derived_lock.is_symlink():
        raise AdapterError(f"derived lock must be a regular file: {derived_lock}")
    if args.device_index != 0:
        raise AdapterError("sLLM Qwen3.8 direct target requires logical device index 0")
    if args.chunk_size <= 0 or args.chunk_size > MAX_TOKENS:
        raise AdapterError(f"--chunk-size must be in 1..={MAX_TOKENS}")
    lock, lock_bytes = read_json(model_lock)
    if (
        lock.get("schema_version") != "model-lock-v1"
        or lock.get("model", {}).get("repo_id") != MODEL_REPO
        or lock.get("fingerprint") != MODEL_FINGERPRINT
    ):
        raise AdapterError("model lock is not the reviewed Qwen3.8-27B identity")
    source_info = inspect_qwen38_source(source_root)
    input_manifest, manifest_bytes = read_input_manifest(manifest_path)
    derived, derived_bytes = read_json(derived_lock)
    if (
        derived.get("schema_version") != "derived-gguf-lock-v1"
        or MODEL_FINGERPRINT not in derived.get("source_lock_fingerprints", [])
    ):
        raise AdapterError("derived lock does not authenticate the Qwen3.8 model lock")
    tensor_mode = (
        derived.get("converter", {})
        .get("effective_config", {})
        .get("tensor_mode", "")
    )
    if not isinstance(tensor_mode, str) or args.kind.upper() not in tensor_mode.upper():
        raise AdapterError(
            f"derived GGUF tensor mode {tensor_mode!r} does not match --kind {args.kind}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    worker = args.worker.resolve() if args.worker else next((p for p in worker_candidates(repo) if p.is_file()), None)
    execution: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA,
        "state": "PREFLIGHT",
        "engine": "sLLM",
        "model": MODEL_REPO,
        "model_fingerprint": MODEL_FINGERPRINT,
        "model_lock": str(model_lock),
        "model_lock_sha256": sha256_bytes(lock_bytes),
        "kind": args.kind,
        "kv": args.kv,
        "chunk_size": args.chunk_size,
        "source": source_info,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "gguf": str(gguf),
        "derived_lock": str(derived_lock),
        "derived_lock_sha256": sha256_bytes(derived_bytes),
        "target": args.target,
        "device_index": args.device_index,
        "worker": str(worker) if worker else None,
        "blockers": [],
    }
    execution_path = output_dir / "execution.json"

    def save() -> None:
        execution_path.write_text(json.dumps(execution, indent=2, ensure_ascii=False) + "\n")

    if worker is None:
        execution["state"] = "BLOCKED"
        execution["blockers"] = [{
            "id": "sllm-qwen38-mx-worker-missing",
            "status": "unsupported",
            "detail": "Build target sllm-qwen38-mx-kld-dump before running this lane.",
        }]
        save()
        print(json.dumps(execution, ensure_ascii=False))
        return 2
    if not worker.is_file() or worker.is_symlink():
        raise AdapterError(f"worker must be a regular file: {worker}")
    command = [
        str(worker),
        "--model-lock", str(model_lock),
        "--model", str(gguf),
        "--derived-lock", str(derived_lock),
        "--manifest", str(manifest_path),
        "--output-dir", str(output_dir),
        "--target", args.target,
        "--device-index", str(args.device_index),
        "--kv", args.kv,
        "--chunk-size", str(args.chunk_size),
    ]
    execution["command"] = command
    execution["started_unix"] = time.time()
    save()
    stdout_path = output_dir / "worker.stdout"
    stderr_path = output_dir / "worker.stderr"
    environment = os.environ.copy()
    library_paths = ["/opt/rocm/core-7.14/lib", str(worker.parent)]
    if environment.get("LD_LIBRARY_PATH"):
        library_paths.append(environment["LD_LIBRARY_PATH"])
    environment["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(library_paths))
    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            completed = subprocess.run(command, stdout=stdout, stderr=stderr, env=environment, check=False)
        execution["returncode"] = completed.returncode
        if completed.returncode != 0:
            execution["state"] = "FAILED"
            execution["error_tail"] = stderr_path.read_text(errors="replace")[-4000:]
        else:
            output_contract(output_dir, input_manifest, args.chunk_size)
            execution["state"] = "PASS"
            execution["output_manifest"] = str(output_dir / "manifest.json")
            execution["output_manifest_sha256"] = sha256_file(output_dir / "manifest.json")
    except (OSError, ValueError, AdapterError) as error:
        execution["state"] = "FAILED"
        execution["error"] = str(error)
    finally:
        execution["ended_unix"] = time.time()
        execution["elapsed_seconds"] = execution["ended_unix"] - execution["started_unix"]
        save()
    print(json.dumps(execution, ensure_ascii=False))
    return 0 if execution["state"] == "PASS" else 1


def main() -> int:
    args = build_parser().parse_args()
    try:
        return run(args)
    except (AdapterError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"qwen38_kld_sllm_mx.py: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
