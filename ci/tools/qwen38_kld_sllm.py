#!/usr/bin/env python3
"""Run the sLLM Qwen3.8 full-vocabulary KLD dump adapter.

The Rust adapter owns model execution and emits one little-endian FP32 file
per manifest case.  This driver keeps the run contract and failure evidence
on disk so a failed format or KV variant can be recorded while a larger
matrix continues.  It intentionally does not calculate KLD: the comparison
driver can consume these raw rows together with BF16/vLLM/llama.cpp dumps.

Manifest format::

    {"schema_version": "qwen38-kld-manifest-v1",
     "cases": [{"id": "short", "token_ids": [1, 2, 3]}]}

``positions`` is optional per case.  Omitting it stores every input position;
providing it stores only the listed zero-based positions while still running
the complete teacher-forced prefix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import shlex
import subprocess
import sys
import time
from typing import Any


SCHEMA_VERSION = "sllm-qwen38-kld-driver-v1"
MANIFEST_SCHEMA = "qwen38-kld-manifest-v1"
VOCAB_SIZE = 248_320
MAX_CASES = 256
MAX_CASE_TOKENS = 262_144
SERIAL_EXECUTION_MODE = "prefill_1_then_serial_decode_with_last_logits"
BLOCK_EXECUTION_MODE = "prefill_1_then_decode_block_with_mtp_state_and_logits_resolve"
SERIAL_LOGIT_STORAGE_PRECISION = "FP32 last_logits API"
BLOCK_LOGIT_STORAGE_PRECISION = "initial FP32 last_logits; subsequent BF16 widened to FP32"


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def read_manifest(path: pathlib.Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    if not raw or len(raw) > 16 * 1024 * 1024:
        raise ValueError("manifest bytes must be in 1..=16777216")
    value = json.loads(raw)
    if not isinstance(value, dict) or value.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError(f"manifest schema must be {MANIFEST_SCHEMA}")
    cases = value.get("cases")
    if not isinstance(cases, list) or not 0 < len(cases) <= MAX_CASES:
        raise ValueError(f"manifest cases must be in 1..={MAX_CASES}")
    ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict) or set(case) - {"id", "token_ids", "positions"}:
            raise ValueError("manifest cases contain an unknown field or are not objects")
        case_id = case.get("id")
        tokens = case.get("token_ids")
        if not isinstance(case_id, str) or not case_id or case_id in ids:
            raise ValueError("case ids must be non-empty and unique strings")
        ids.add(case_id)
        if (
            not isinstance(tokens, list)
            or not 0 < len(tokens) <= MAX_CASE_TOKENS
            or any(not isinstance(token, int) or isinstance(token, bool) or not 0 <= token < VOCAB_SIZE for token in tokens)
        ):
            raise ValueError(f"case {case_id} token_ids are outside the model vocabulary")
        positions = case.get("positions")
        if positions is not None:
            if (
                not isinstance(positions, list)
                or not positions
                or any(not isinstance(position, int) or isinstance(position, bool) for position in positions)
                or len(set(positions)) != len(positions)
                or any(position < 0 or position >= len(tokens) for position in positions)
            ):
                raise ValueError(f"case {case_id} positions are invalid")
    return raw, value


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--binary", required=True, type=pathlib.Path, help="sllm-qwen38-kld-dump executable")
    result.add_argument("--model-root", required=True, type=pathlib.Path)
    result.add_argument("--manifest", required=True, type=pathlib.Path)
    result.add_argument("--output-dir", required=True, type=pathlib.Path)
    result.add_argument("--target", choices=("gfx1030", "gfx1201"), required=True)
    result.add_argument("--device-index", type=int, default=0)
    result.add_argument("--kv", choices=("fp16", "kv-mxfp8-e4", "kv-mxfp8-e5"), required=True)
    result.add_argument("--chunk-size", type=int, default=1)
    result.add_argument("--timeout-seconds", type=float, default=86_400.0)
    result.add_argument("--cwd", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[2])
    return result


def validate_output_manifest(
    output_dir: pathlib.Path,
    input_manifest: dict[str, Any],
    chunk_size: int = 1,
) -> dict[str, Any]:
    """Validate the stable cross-engine manifest emitted by the Rust worker."""
    path = output_dir / "manifest.json"
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"sLLM output manifest is missing: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("engine") != "sLLM" or value.get("model") != "Qwen3.8-27B":
        raise ValueError("sLLM output manifest has an invalid engine")
    if not 0 < chunk_size <= MAX_CASE_TOKENS:
        raise ValueError(f"chunk_size must be in 1..={MAX_CASE_TOKENS}")
    if value.get("chunk_size") != chunk_size:
        raise ValueError("sLLM output chunk_size differs from the requested chunk size")
    expected_mode = SERIAL_EXECUTION_MODE if chunk_size == 1 else BLOCK_EXECUTION_MODE
    expected_precision = SERIAL_LOGIT_STORAGE_PRECISION if chunk_size == 1 else BLOCK_LOGIT_STORAGE_PRECISION
    if value.get("execution_mode") != expected_mode:
        raise ValueError("sLLM output execution mode differs from the requested chunk mode")
    if value.get("logit_storage_precision") != expected_precision:
        raise ValueError("sLLM output logit storage precision differs from the requested chunk mode")
    cases = value.get("cases")
    if not isinstance(cases, list) or len(cases) != len(input_manifest["cases"]):
        raise ValueError("sLLM output manifest case count differs from input")
    expected_ids = [case["id"] for case in input_manifest["cases"]]
    observed_ids = [case.get("id") for case in cases if isinstance(case, dict)]
    if observed_ids != expected_ids:
        raise ValueError("sLLM output manifest case order or IDs differ from input")
    for item, source in zip(cases, input_manifest["cases"]):
        if not isinstance(item, dict):
            raise ValueError("sLLM output manifest case is not an object")
        rows, vocab = item.get("shape", [None, None])
        positions = item.get("positions")
        if vocab != VOCAB_SIZE or not isinstance(rows, int) or rows != len(positions or []):
            raise ValueError(f"sLLM output manifest shape is invalid for {source['id']}")
        expected_positions = source.get("positions") or list(range(len(source["token_ids"])))
        if positions != sorted(expected_positions):
            raise ValueError(f"sLLM output manifest positions differ for {source['id']}")
        expected_chunks = [1]
        remaining = max(0, len(source["token_ids"]) - 1)
        while remaining:
            count = min(chunk_size, remaining)
            expected_chunks.append(count)
            remaining -= count
        if item.get("actual_chunk_sizes") != expected_chunks:
            raise ValueError(f"sLLM output chunk sizes differ for {source['id']}")
        if item.get("nonfinite_count") != 0:
            raise ValueError(f"sLLM reported non-finite logits for {source['id']}")
        filename = item.get("logits_file")
        if not isinstance(filename, str) or pathlib.PurePath(filename).name != filename or filename != f"{source['id']}.f32":
            raise ValueError(f"sLLM output filename is unsafe or mismatched for {source['id']}")
        raw_path = output_dir / filename
        expected_bytes = rows * vocab * 4
        if not raw_path.is_file() or raw_path.is_symlink() or raw_path.stat().st_size != expected_bytes:
            raise ValueError(f"sLLM raw logits size is invalid for {source['id']}")
    return value


def run(arguments: argparse.Namespace) -> int:
    binary = arguments.binary.resolve()
    model_root = arguments.model_root.resolve()
    manifest = arguments.manifest.resolve()
    output_dir = arguments.output_dir.resolve()
    cwd = arguments.cwd.resolve()
    if arguments.device_index != 0:
        raise ValueError("sLLM Qwen3.8 direct artifact requires --device-index 0")
    if not 0 < arguments.chunk_size <= MAX_CASE_TOKENS:
        raise ValueError(f"--chunk-size must be in 1..={MAX_CASE_TOKENS}")
    for path, label in ((binary, "binary"), (model_root, "model root"), (manifest, "manifest"), (cwd, "cwd")):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    manifest_bytes, input_manifest = read_manifest(manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / "adapter.stdout"
    stderr_path = output_dir / "adapter.stderr"
    execution_path = output_dir / "execution.json"
    command = [
        str(binary),
        "--model-root",
        str(model_root),
        "--manifest",
        str(manifest),
        "--output-dir",
        str(output_dir),
        "--target",
        arguments.target,
        "--device-index",
        str(arguments.device_index),
        "--kv",
        arguments.kv,
        "--chunk-size",
        str(arguments.chunk_size),
    ]
    environment = os.environ.copy()
    environment.setdefault("LD_LIBRARY_PATH", "/opt/rocm/lib")
    started = time.time()
    execution: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "state": "RUNNING",
        "model_root": str(model_root),
        "manifest": str(manifest),
        "manifest_sha256": f"sha256:{hashlib.sha256(manifest_bytes).hexdigest()}",
        "output_dir": str(output_dir),
        "target": arguments.target,
        "device_index": arguments.device_index,
        "kv_cache_encoding": arguments.kv,
        "chunk_size": arguments.chunk_size,
        "binary": str(binary),
        "binary_sha256": sha256_file(binary),
        "command": [str(item) for item in command],
        "command_shell": shlex.join(command),
        "started_unix": started,
    }

    def save() -> None:
        execution_path.write_text(json.dumps(execution, indent=2, ensure_ascii=False) + "\n")

    save()
    try:
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            completed = subprocess.run(
                command,
                cwd=cwd,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                timeout=arguments.timeout_seconds,
                check=False,
            )
        execution["returncode"] = completed.returncode
        report_path = output_dir / "report.json"
        if completed.returncode == 0 and stdout_path.stat().st_size:
            report = json.loads(stdout_path.read_text())
            validate_output_manifest(output_dir, input_manifest, arguments.chunk_size)
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
            execution["report"] = str(report_path)
            execution["output_manifest"] = str(output_dir / "manifest.json")
            execution["output_manifest_sha256"] = sha256_file(output_dir / "manifest.json")
            execution["state"] = "PASS"
        else:
            execution["state"] = "FAIL"
            if stderr_path.stat().st_size:
                execution["error_tail"] = stderr_path.read_text(errors="replace")[-2000:]
    except subprocess.TimeoutExpired as error:
        execution["state"] = "TIMEOUT"
        execution["timeout_seconds"] = arguments.timeout_seconds
        execution["error"] = str(error)
    finally:
        execution["ended_unix"] = time.time()
        execution["elapsed_seconds"] = execution["ended_unix"] - started
        save()
    print(json.dumps(execution, ensure_ascii=False))
    return 0 if execution["state"] == "PASS" else 1


def main() -> int:
    arguments = parser().parse_args()
    try:
        return run(arguments)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"qwen38_kld_sllm.py: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
