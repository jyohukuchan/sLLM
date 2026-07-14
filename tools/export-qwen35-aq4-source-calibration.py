#!/usr/bin/env python3
"""Capture a bounded full-vector BF16 Qwen3.5-9B source calibration.

This tool is intentionally separate from ``export-qwen35-aq4-source-oracle.py``.
The older oracle stores bounded JSON samples and remains immutable.  This
calibration artifact stores one final, normalized hidden row and one raw
pre-softmax logit row per observation in little-endian F32 sidecars.  The
model is loaded once and vectors are written row-by-row; no sequence by
vocabulary matrix is retained.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import shutil
import stat
import struct
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen35_aq4_p2_oracle as legacy_oracle  # noqa: E402


SCHEMA = "ullm.qwen35_aq4_source_calibration.v1"
ORACLE_KIND = "independent_source_full"
CASES_SCHEMA = "ullm.qwen35_aq4_source_calibration_cases.v1"
HIDDEN_SIZE = 4096
VOCAB_SIZE = 248320
TOP_K = 10
DEFAULT_CHUNK_ELEMENTS = 65536
MAX_CASE_FILE_BYTES = 4 * 1024 * 1024
MAX_CASES = 8192
MAX_STEPS = 128
MAX_ROWS = 16384
MAX_ROWS_FILE_BYTES = 64 * 1024 * 1024
MAX_ROW_LINE_BYTES = 64 * 1024
MIN_AVAILABLE_HEADROOM = 2.0
F32_BYTES = 4
ROW_BYTES = (HIDDEN_SIZE + VOCAB_SIZE) * F32_BYTES
TOKENIZER_FILES = legacy_oracle.TOKENIZER_FILES


class CalibrationError(ValueError):
    pass


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise CalibrationError(f"duplicate JSON key: {key}")
        value[key] = child
    return value


def reject_nonfinite(value: str) -> None:
    raise CalibrationError(f"non-finite JSON number: {value}")


def _stat_identity_info(info: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_nlink, info.st_mode)


def _stat_identity(path: Path) -> tuple[int, int, int, int, int, int, int]:
    return _stat_identity_info(os.lstat(path))


def regular_file(path: Path, label: str, *, max_bytes: int | None = None) -> Path:
    ensure_no_symlink_ancestors(path, label)
    try:
        info = path.lstat()
    except OSError as error:
        raise CalibrationError(f"{label} is unavailable: {error}") from error
    if not stat.S_ISREG(info.st_mode):
        raise CalibrationError(f"{label} must be a regular file")
    if info.st_nlink != 1:
        raise CalibrationError(f"{label} must have exactly one hard link")
    if max_bytes is not None and info.st_size > max_bytes:
        raise CalibrationError(f"{label} exceeds {max_bytes} bytes")
    return path


@contextmanager
def stable_fd(path: Path, label: str, *, max_bytes: int | None = None):
    regular_file(path, label, max_bytes=max_bytes)
    before = os.lstat(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as error:
        raise CalibrationError(f"{label} cannot be opened safely: {error}") from error
    try:
        if _stat_identity_info(os.fstat(fd)) != _stat_identity_info(before):
            raise CalibrationError(f"{label} changed while being opened")
        yield fd, before
        if _stat_identity_info(os.fstat(fd)) != _stat_identity_info(before) or _stat_identity(path) != _stat_identity_info(before):
            raise CalibrationError(f"{label} changed while being read")
    finally:
        os.close(fd)


def read_fd_all(fd: int, max_bytes: int, label: str) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        raw = os.read(fd, min(1024 * 1024, max_bytes + 1 - total))
        if not raw:
            break
        chunks.append(raw)
        total += len(raw)
        if total > max_bytes:
            raise CalibrationError(f"{label} exceeds {max_bytes} bytes")
    return b"".join(chunks)


def read_stable_json(path: Path, label: str, *, max_bytes: int = MAX_CASE_FILE_BYTES) -> Any:
    with stable_fd(path, label, max_bytes=max_bytes) as (fd, _):
        raw = read_fd_all(fd, max_bytes, label)
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicate_keys, parse_constant=reject_nonfinite)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CalibrationError(f"invalid {label}: {error}") from error


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file_stable(path: Path, label: str, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with stable_fd(path, label) as (fd, _):
        while chunk := os.read(fd, chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_no_symlink_ancestors(path: Path, label: str) -> None:
    absolute = path.absolute()
    components = [Path(absolute.anchor), *(Path(absolute.anchor, *absolute.parts[1:index]) for index in range(1, len(absolute.parts) + 1))]
    for index, component in enumerate(components):
        try:
            info = os.lstat(component)
        except FileNotFoundError:
            if all(not candidate.exists() for candidate in components[index:]):
                return
            raise CalibrationError(f"{label} path component is unavailable: {component}")
        except OSError as error:
            raise CalibrationError(f"{label} path component is unavailable: {error}") from error
        if stat.S_ISLNK(info.st_mode):
            raise CalibrationError(f"{label} path component is a symlink: {component}")


def integer(value: Any, label: str, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or (maximum is not None and value > maximum):
        suffix = f" <= {maximum}" if maximum is not None else ""
        raise CalibrationError(f"{label} must be an integer >= {minimum}{suffix}")
    return value


def finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise CalibrationError(f"{label} must be finite")
    return float(value)


def load_cases(path: Path) -> list[dict[str, Any]]:
    value = read_stable_json(path, "calibration cases")
    if not isinstance(value, dict) or value.get("schema_version") != CASES_SCHEMA or set(value) != {"schema_version", "cases"}:
        raise CalibrationError(f"calibration cases must use {CASES_SCHEMA} with exact fields")
    raw_cases = value["cases"]
    if not isinstance(raw_cases, list) or not raw_cases or len(raw_cases) > MAX_CASES:
        raise CalibrationError(f"calibration cases must contain 1..{MAX_CASES} cases")
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    rows = 0
    for index, raw in enumerate(raw_cases):
        required = {"case_id", "prompt_token_ids", "step_count"}
        allowed = required | {"semantic_input_id", "observation"}
        if not isinstance(raw, dict) or not required <= set(raw) or set(raw) - allowed:
            raise CalibrationError(f"cases[{index}] fields differ")
        case_id = raw.get("case_id")
        if not isinstance(case_id, str) or not case_id or len(case_id) > 128 or case_id in seen:
            raise CalibrationError(f"cases[{index}].case_id must be unique and bounded")
        seen.add(case_id)
        tokens = raw.get("prompt_token_ids")
        if not isinstance(tokens, list) or not tokens or len(tokens) > 4096:
            raise CalibrationError(f"cases[{index}].prompt_token_ids is invalid")
        normalized_tokens = [integer(token, f"cases[{index}].prompt_token_ids", 0, VOCAB_SIZE - 1) for token in tokens]
        step_count = integer(raw.get("step_count"), f"cases[{index}].step_count", 1, MAX_STEPS)
        rows += step_count
        if rows > MAX_ROWS:
            raise CalibrationError(f"calibration rows exceed {MAX_ROWS}")
        semantic = raw.get("semantic_input_id", case_id)
        observation = raw.get("observation", "first_token")
        if not isinstance(semantic, str) or not semantic or len(semantic) > 128:
            raise CalibrationError(f"cases[{index}].semantic_input_id is invalid")
        if not isinstance(observation, str) or not observation or len(observation) > 64:
            raise CalibrationError(f"cases[{index}].observation is invalid")
        cases.append({"case_id": case_id, "prompt_token_ids": normalized_tokens, "step_count": step_count, "semantic_input_id": semantic, "observation": observation})
    return cases


def checkpoint_bytes(model_dir: Path) -> int:
    index = read_stable_json(model_dir / "model.safetensors.index.json", "source weight index", max_bytes=16 * 1024 * 1024)
    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict) or not index["weight_map"]:
        raise CalibrationError("source weight index has no weight_map")
    total = 0
    for name in sorted(set(index["weight_map"].values())):
        if not isinstance(name, str) or not name:
            raise CalibrationError("source weight index contains an invalid shard")
        path = legacy_oracle.safe_relative(model_dir, name, "source checkpoint shard")
        regular_file(path, "source checkpoint shard")
        total += os.lstat(path).st_size
    return total


def memory_preflight(model_dir: Path) -> dict[str, Any]:
    total = checkpoint_bytes(model_dir)
    meminfo: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        fields = rest.split()
        if key in {"MemTotal", "MemAvailable"} and fields:
            meminfo[key] = int(fields[0]) * 1024
    available = meminfo.get("MemAvailable", 0)
    required = int(math.ceil(total * MIN_AVAILABLE_HEADROOM))
    return {"checkpoint_bytes": total, "mem_total_bytes": meminfo.get("MemTotal"), "mem_available_bytes": available, "required_headroom_bytes": required, "headroom_factor": MIN_AVAILABLE_HEADROOM, "status": "passed" if available >= required else "blocked"}


def disk_preflight(parent: Path, expected_rows: int) -> dict[str, Any]:
    expected_bytes = expected_rows * ROW_BYTES
    required = int(math.ceil(expected_bytes * 1.2))
    usage = shutil.disk_usage(parent)
    return {"expected_vector_bytes": expected_bytes, "required_free_bytes": required, "free_bytes": usage.free, "status": "passed" if usage.free >= required else "blocked"}


def f32_bytes(tensor: torch.Tensor) -> bytes:
    tensor = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().flatten()
    try:
        import numpy as np
    except ImportError as error:
        raise CalibrationError(f"numpy is required for f32le sidecars: {error}") from error
    values = tensor.numpy()
    return values.astype("<f4", copy=False).tobytes(order="C")


def write_vector(handle: Any, tensor: torch.Tensor, chunk_elements: int) -> tuple[int, str, int]:
    flat = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().flatten()
    digest = hashlib.sha256()
    nonfinite = int((~torch.isfinite(flat)).sum().item())
    start = handle.tell()
    for chunk in flat.split(chunk_elements):
        encoded = f32_bytes(chunk)
        handle.write(encoded)
        digest.update(encoded)
    return start, digest.hexdigest(), nonfinite


def topk(logits: torch.Tensor, count: int) -> list[dict[str, Any]]:
    values = logits.detach().to(device="cpu", dtype=torch.float32).flatten()
    indices = torch.argsort(values, descending=True, stable=True)[:count]
    return [{"token_id": int(token), "logit": float(values[token])} for token in indices.tolist()]


def publish_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise CalibrationError("renameat2(RENAME_NOREPLACE) is unavailable")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise CalibrationError(f"refusing to overwrite existing output: {destination}")
        raise CalibrationError(f"exclusive output publish failed: {os.strerror(error)}")
    parent_fd = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def source_identity(model_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    inspected = legacy_oracle.inspect_source_model(model_dir)
    files = sorted([inspected["config"], inspected["weight_index"], *inspected["weight_shards"]], key=lambda item: item["file"])
    source_checkpoint = {"aggregate_sha256": legacy_oracle.canonical_sha256(files), "dtype": inspected["dtype"], "files": files, "root": inspected["root"]}
    tokenizer_files = sorted(inspected["tokenizer_files"], key=lambda item: item["file"])
    tokenizer = {"aggregate_sha256": legacy_oracle.canonical_sha256(tokenizer_files), "files": tokenizer_files, "root": inspected["root"]}
    identity = {"artifact": {"package_manifest_sha256": None, "artifact_manifest_sha256": None}, "model_id": inspected["model_id"], "model_revision": inspected["revision"], "source_checkpoint": source_checkpoint, "tokenizer": tokenizer, "hidden_size": HIDDEN_SIZE, "vocab_size": VOCAB_SIZE}
    return identity, inspected


def validate_legacy_identity(legacy_root: Path, identity: dict[str, Any]) -> dict[str, Any]:
    manifest = legacy_oracle.validate_manifest(legacy_root, expected_kind="independent_source")
    if manifest.get("schema_version") != legacy_oracle.SOURCE_SCHEMA or manifest.get("status") not in {"available", "fixture"}:
        raise CalibrationError("legacy oracle is not source-oracle-v1")
    old = manifest["identity"]
    for key in ("model_id", "model_revision"):
        if old.get(key) != identity.get(key):
            raise CalibrationError(f"legacy source identity differs: {key}")
    if old["tokenizer"]["aggregate_sha256"] != identity["tokenizer"]["aggregate_sha256"] or old["source_checkpoint"]["aggregate_sha256"] != identity["source_checkpoint"]["aggregate_sha256"]:
        raise CalibrationError("legacy source checkpoint/tokenizer identity differs")
    return manifest


def _row_map(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    with stable_fd(path, "row index", max_bytes=MAX_ROWS_FILE_BYTES) as (fd, _):
        pending = b""
        line_number = 0
        while True:
            chunk = os.read(fd, 64 * 1024)
            if not chunk:
                break
            pending += chunk
            while b"\n" in pending:
                raw, pending = pending.split(b"\n", 1)
                line_number += 1
                if not raw or len(raw) > MAX_ROW_LINE_BYTES:
                    raise CalibrationError(f"row index line {line_number} is empty or oversized")
                try:
                    row = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicate_keys, parse_constant=reject_nonfinite)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise CalibrationError(f"invalid row index line {line_number}: {error}") from error
                if not isinstance(row, dict) or not isinstance(row.get("case_id"), str) or not isinstance(row.get("step"), int):
                    raise CalibrationError(f"row index line {line_number} is invalid")
                key = (row["case_id"], row["step"])
                if key in result:
                    raise CalibrationError("row index contains duplicate case/step")
                result[key] = row
                if len(result) > MAX_ROWS:
                    raise CalibrationError("row index exceeds record bound")
            if len(pending) > MAX_ROW_LINE_BYTES:
                raise CalibrationError("row index line exceeds byte bound")
        if pending:
            line_number += 1
            try:
                row = json.loads(pending.decode("utf-8"), object_pairs_hook=reject_duplicate_keys, parse_constant=reject_nonfinite)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise CalibrationError(f"invalid row index line {line_number}: {error}") from error
            if not isinstance(row, dict) or not isinstance(row.get("case_id"), str) or not isinstance(row.get("step"), int):
                raise CalibrationError(f"row index line {line_number} is invalid")
            key = (row["case_id"], row["step"])
            if key in result:
                raise CalibrationError("row index contains duplicate case/step")
            result[key] = row
            if len(result) > MAX_ROWS:
                raise CalibrationError("row index exceeds record bound")
    return result


def read_values(handle: Any, offset: int, elements: int) -> list[float]:
    raw = os.pread(handle, elements * F32_BYTES, offset)
    if len(raw) != elements * F32_BYTES:
        raise CalibrationError("vector sidecar is shorter than row index")
    return list(struct.unpack(f"<{elements}f", raw))


def legacy_cross_check(legacy_root: Path, legacy_manifest: dict[str, Any], row_index_path: Path, hidden_path: Path, logits_path: Path) -> dict[str, Any]:
    rows = _row_map(row_index_path)
    checked = 0
    hidden_max = 0.0
    logit_max = 0.0
    with stable_fd(hidden_path, "generated hidden sidecar") as (hidden, _), stable_fd(logits_path, "generated logits sidecar") as (logits, _):
        for old in legacy_oracle.payload_records(legacy_root, legacy_manifest):
            key = (old["case_id"], old["step"])
            row = rows.get(key)
            if row is None:
                raise CalibrationError(f"full calibration misses legacy row {key}")
            hidden_values = read_values(hidden, int(row["hidden"]["offset_bytes"]), HIDDEN_SIZE)
            logit_values = read_values(logits, int(row["logits"]["offset_bytes"]), VOCAB_SIZE)
            for index, expected in zip(old["hidden_sample"]["indices"], old["hidden_sample"]["values"]):
                hidden_max = max(hidden_max, abs(hidden_values[index] - float(expected)))
            for index, expected in zip(old["logit_sample"]["indices"], old["logit_sample"]["values"]):
                logit_max = max(logit_max, abs(logit_values[index] - float(expected)))
            if row["greedy_token_id"] != old["greedy_token_id"] or row["topk"] != old["topk"]:
                raise CalibrationError(f"legacy top-k/greedy differs for {key}")
            checked += 1
    return {"status": "passed", "legacy_manifest_sha256": sha256_file_stable(legacy_root / "manifest.json", "legacy manifest"), "legacy_payload_sha256": legacy_manifest["payload"]["sha256"], "row_count": checked, "hidden_sample_max_abs_diff": hidden_max, "logit_sample_max_abs_diff": logit_max}


def capture_model(model_dir: Path, cases: list[dict[str, Any]], temporary: Path, *, chunk_elements: int, top_k_count: int, threads: int) -> dict[str, Any]:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if torch.cuda.is_available():
        raise CalibrationError("GPU is visible; source calibration is CPU-only")
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(threads)
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as error:
        raise CalibrationError(f"transformers is unavailable: {error}") from error
    started = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, dtype=torch.bfloat16, low_cpu_mem_usage=False, device_map=None)
    model.eval()
    hidden_path = temporary / "vectors" / "hidden.f32le"
    logits_path = temporary / "vectors" / "logits.f32le"
    rows_path = temporary / "rows.jsonl"
    hidden_path.parent.mkdir(parents=True, exist_ok=False)
    row_count = 0
    nonfinite_rows = 0
    try:
        with hidden_path.open("wb") as hidden_out, logits_path.open("wb") as logits_out, rows_path.open("w", encoding="utf-8") as rows_out:
            for case in cases:
                past = None
                input_ids = list(case["prompt_token_ids"])
                try:
                    for step in range(case["step_count"]):
                        token_tensor = torch.tensor([input_ids], dtype=torch.long, device="cpu")
                        with torch.inference_mode():
                            base = model.model(input_ids=token_tensor, past_key_values=past, use_cache=True, return_dict=True)
                            hidden_tensor = base.last_hidden_state[:, -1, :]
                            logits_tensor = model.lm_head(hidden_tensor.unsqueeze(1))[:, -1, :]
                        hidden_offset, hidden_sha, hidden_nonfinite = write_vector(hidden_out, hidden_tensor, chunk_elements)
                        logits_offset, logits_sha, logits_nonfinite = write_vector(logits_out, logits_tensor, chunk_elements)
                        vector_finite = hidden_nonfinite == 0 and logits_nonfinite == 0
                        top = topk(logits_tensor, top_k_count) if vector_finite else None
                        greedy = top[0]["token_id"] if top is not None else None
                        row = {"case_id": case["case_id"], "step": step, "semantic_input_id": case["semantic_input_id"], "observation": case["observation"], "input_token_ids_sha256": legacy_oracle.canonical_token_ids_hash(input_ids), "hidden": {"offset_bytes": hidden_offset, "bytes": HIDDEN_SIZE * F32_BYTES, "elements": HIDDEN_SIZE, "dtype": "f32", "endianness": "little", "sha256": hidden_sha, "nonfinite_count": hidden_nonfinite}, "logits": {"offset_bytes": logits_offset, "bytes": VOCAB_SIZE * F32_BYTES, "elements": VOCAB_SIZE, "dtype": "f32", "endianness": "little", "sha256": logits_sha, "nonfinite_count": logits_nonfinite}, "greedy_token_id": greedy, "topk": top, "finite": vector_finite}
                        nonfinite_rows += int(not row["finite"])
                        rows_out.write(json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n")
                        rows_out.flush()
                        row_count += 1
                        if not vector_finite:
                            break
                        input_ids = [greedy]
                        past = getattr(base, "past_key_values", None)
                        del token_tensor, base, hidden_tensor, logits_tensor, top
                    if nonfinite_rows:
                        break
                finally:
                    del past
                    gc.collect()
    finally:
        del model
        gc.collect()
    return {"row_count": row_count, "nonfinite_rows": nonfinite_rows, "elapsed_seconds": time.monotonic() - started}


def export(args: argparse.Namespace) -> dict[str, Any]:
    ensure_no_symlink_ancestors(args.model_dir, "model")
    ensure_no_symlink_ancestors(args.cases, "cases")
    ensure_no_symlink_ancestors(args.output.parent, "output")
    if args.output.exists() or os.path.lexists(args.output):
        raise CalibrationError(f"refusing to overwrite existing output: {args.output}")
    cases = load_cases(args.cases)
    identity, _inspected = source_identity(args.model_dir)
    memory = memory_preflight(args.model_dir)
    if memory["status"] != "passed":
        raise CalibrationError(f"CPU memory preflight failed: available={memory['mem_available_bytes']} required={memory['required_headroom_bytes']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ensure_no_symlink_ancestors(args.output.parent, "output")
    disk = disk_preflight(args.output.parent, sum(case["step_count"] for case in cases))
    if disk["status"] != "passed":
        raise CalibrationError(f"disk preflight failed: free={disk['free_bytes']} required={disk['required_free_bytes']}")
    if args.legacy_oracle is None:
        raise CalibrationError("--legacy-oracle is required for parent sampled identity/cross-check")
    legacy_root = args.legacy_oracle
    legacy_manifest = validate_legacy_identity(legacy_root, identity)
    temporary = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.incomplete-", dir=args.output.parent))
    try:
        run = capture_model(args.model_dir, cases, temporary, chunk_elements=args.chunk_elements, top_k_count=args.top_k, threads=args.threads)
        if run["nonfinite_rows"] == 0:
            compatibility = legacy_cross_check(legacy_root, legacy_manifest, temporary / "rows.jsonl", temporary / "vectors" / "hidden.f32le", temporary / "vectors" / "logits.f32le")
        else:
            compatibility = {"status": "blocked", "legacy_manifest_sha256": sha256_file_stable(legacy_root / "manifest.json", "legacy manifest"), "legacy_payload_sha256": legacy_manifest["payload"]["sha256"], "row_count": 0, "hidden_sample_max_abs_diff": 0.0, "logit_sample_max_abs_diff": 0.0}
        manifest = {"schema_version": SCHEMA, "oracle_kind": ORACLE_KIND, "status": "available" if run["nonfinite_rows"] == 0 else "blocked", "evidence_class": "production" if run["nonfinite_rows"] == 0 else "blocked", "usable_as_source_evidence": run["nonfinite_rows"] == 0, "promotion_eligible": False, "created_utc": legacy_oracle.utc_now(), "identity": identity, "parent_sampled_oracle": {"path": str((legacy_root / "manifest.json").resolve(strict=True)), "manifest_sha256": compatibility["legacy_manifest_sha256"], "schema_version": legacy_oracle.SOURCE_SCHEMA}, "vector_contract": {"hidden_shape": [HIDDEN_SIZE], "logits_shape": [VOCAB_SIZE], "dtype": "f32", "endianness": "little", "layout": "flat", "chunk_elements": args.chunk_elements, "row_bytes": ROW_BYTES, "semantic_hidden": "final_rmsnorm_hidden_used_by_lm_head", "semantic_logits": "raw_pre_softmax_lm_head_logits"}, "limits": {"max_case_file_bytes": MAX_CASE_FILE_BYTES, "max_cases": MAX_CASES, "max_rows": MAX_ROWS, "max_steps": MAX_STEPS}, "cases": {"path": str(args.cases.resolve(strict=True)), "sha256": sha256_file_stable(args.cases, "calibration cases"), "case_count": len(cases), "row_count": run["row_count"]}, "files": {"rows": "rows.jsonl", "hidden": "vectors/hidden.f32le", "logits": "vectors/logits.f32le"}, "runtime": {"runtime": "transformers.AutoModelForCausalLM", "transformers": package_version("transformers"), "torch": package_version("torch"), "safetensors": package_version("safetensors"), "python": platform.python_version(), "device": "cpu", "dtype": "bfloat16", "low_cpu_mem_usage": False, "torch_num_threads": args.threads, "torch_num_interop_threads": args.threads, "model_loads": 1, "inference_mode": True, "full_vocab_ranking": run["nonfinite_rows"] == 0, "max_resident_logit_rows": 1, "memory_preflight": memory, "disk_preflight": disk, "run": run}, "legacy_cross_check": compatibility}
        (temporary / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for path in (temporary / "rows.jsonl", temporary / "vectors" / "hidden.f32le", temporary / "vectors" / "logits.f32le"):
            sha256_file_stable(path, f"calibration {path.name}")
        sums = []
        for path in sorted(temporary.rglob("*")):
            if path.is_file() and path.name != "SHA256SUMS":
                sums.append(f"{sha256_file_stable(path, path.name)}  {path.relative_to(temporary).as_posix()}\n")
        (temporary / "SHA256SUMS").write_text("".join(sums), encoding="ascii")
        for path in (temporary / "rows.jsonl", temporary / "vectors" / "hidden.f32le", temporary / "vectors" / "logits.f32le", temporary / "manifest.json", temporary / "SHA256SUMS"):
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        for path in (temporary / "vectors", temporary):
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0))
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        publish_noreplace(temporary, args.output)
        return manifest
    except Exception:
        raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-oracle", type=Path, required=True)
    parser.add_argument("--chunk-elements", type=int, default=DEFAULT_CHUNK_ELEMENTS)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.chunk_elements <= 0 or args.chunk_elements > 1_048_576 or args.top_k <= 0 or args.top_k > 32 or args.threads <= 0 or args.threads > 16:
        parser.error("chunk-elements/top-k/threads are outside bounded limits")
    try:
        result = export(args)
        print(json.dumps(result, ensure_ascii=True, sort_keys=True))
        return 0
    except (CalibrationError, legacy_oracle.OracleError, OSError, RuntimeError, ValueError) as error:
        print(f"Qwen3.5 full source calibration failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
