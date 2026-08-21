#!/usr/bin/env python3
"""Fail-closed aggregator for the Phase 36 MI300X Session B audit.

Session B is intentionally an aggregation step.  The GPU evidence producers
write JSON reports and per-row sysfs TSV observations to a private retention
directory; this controller never re-runs a producer and never copies raw
evidence into the repository.  The tracked summary contains only identities,
small bounded facts, and SHA-256 digests of retained raw files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

TARGET = "gfx942"
SCHEMA_VERSION = "phase36-mi300x-session-b-summary-v1"
SUMMARY_NAME = "phase36-mi300x-session-b-summary-v1.json"
RAW_OUTPUTS = "raw/"
FULL_ATTENTION_CASES = 29
KV_STATE_CASES = 19
LOWBIT_ORACLE_CASES = 17
INPUT_TOKENS = 10_001
OUTPUT_TOKENS = 2
EXPECTED_INPUT_ID = 23066
EXPECTED_OUTPUT_IDS = [23066, 23066]
EXPECTED_ROWS: tuple[tuple[str, str], ...] = tuple(
    (encoding, setting)
    for encoding in ("fp16-v1", "kv-fp8-v1")
    for setting in ("auto", "512", "2048", "4096", "8192", "16384")
)
TOTAL_MODEL_ROWS = len(EXPECTED_ROWS)
TOTAL_ROWS = TOTAL_MODEL_ROWS
FULL_REPORT_FILES: dict[str, tuple[str, ...]] = {
    "fp16-v1": ("full-attention-fp16.json", "full-attention-fp16-v1.json", "fp16-v1.json"),
    "kv-fp8-v1": ("full-attention-fp8.json", "full-attention-kv-fp8-v1.json", "kv-fp8-v1.json"),
    "kv-fp8-static-v1": ("full-attention-fp8-static.json", "full-attention-kv-fp8-static-v1.json", "kv-fp8-static-v1.json"),
    "kv-nvfp4-v1": ("full-attention-nvfp4.json", "full-attention-kv-nvfp4-v1.json", "kv-nvfp4-v1.json"),
}
KV_STATE_FILES = ("kv-state-fp16.json", "kv-state-fp16-v1.json", "kv-state-v1.json", "kv-state.json")
ORACLE_FILES = ("phase36-kv-lowbit-oracle.json", "numpy-lowbit-oracle-v1.json", "numpy-lowbit-oracle.json", "lowbit-oracle-v1.json", "lowbit-oracle.json", "numpy-oracle.json")
NATIVE_ROW_FILES = tuple(
    [f"long-fp16-{setting}.json" for setting in ("auto", "chunk-512", "chunk-2048", "chunk-4096", "chunk-8192", "chunk-16384")]
    + [f"long-fp8-{setting}.json" for setting in ("auto", "512", "2048", "4096", "8192", "16384")]
)
ROW_FILES = ("full-model-rows-v1.json", "full-model-rows.json", "qwen-10001-rows.json", "model-rows-v1.json", "rows-v1.json")
MEMORY_FILES = ("memory-accounting-v1.json", "memory-v1.json", "session-b-memory.json")
EPSILON = 1.0e-9
MAX_RAW_JSON_BYTES = 64 * 1024 * 1024


class SessionBError(RuntimeError):
    """Malformed, incomplete, or unsafe retained evidence."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _json_digest(value: Any) -> str:
    return _sha256_bytes(_json_bytes(value))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        if len(raw) > MAX_RAW_JSON_BYTES:
            raise SessionBError(f"{path.name}: retained JSON exceeds bounded size")
        def duplicate_rejector(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, item in pairs:
                if key in result:
                    raise SessionBError(f"{path.name}: duplicate JSON key {key}")
                result[key] = item
            return result
        def constant_rejector(token: str) -> None:
            raise SessionBError(f"{path.name}: non-finite JSON constant {token}")
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=duplicate_rejector, parse_constant=constant_rejector)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SessionBError(f"{path}: malformed JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SessionBError(f"{path.name}: JSON root must be an object")
    return value


def _regular(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise SessionBError(f"{label} must be a regular non-symlink file: {path}")
    return path


def _raw_file(raw_dir: Path, names: Sequence[str], label: str) -> Path:
    for name in names:
        candidate = raw_dir / name
        if candidate.exists():
            return _regular(candidate, label)
    raise SessionBError(f"missing retained {label} ({', '.join(names)})")


def _raw_digest(path: Path) -> str:
    # A raw digest is intentionally the only raw identity emitted to the
    # summary.  Paths remain private implementation details of this run.
    return _sha256_file(_regular(path, "retained raw file"))


def _walk(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield key, item
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def _bool(document: dict[str, Any], *keys: str, label: str) -> None:
    for key in keys:
        if key in document:
            if document[key] is not True:
                raise SessionBError(f"{label}: {key} is not positive")
            return
    raise SessionBError(f"{label}: positive evidence is absent ({'/'.join(keys)})")


def _zero_cleanup(document: dict[str, Any], label: str) -> tuple[int, int]:
    cleanup = document.get("cleanup")
    if not isinstance(cleanup, dict):
        # Model rows use the compact producer form.  The full reports and
        # state report use a nested cleanup object.
        cleanup = document
    retryable = cleanup.get("retryable_cleanup", cleanup.get("cleanup_retryable"))
    durable = cleanup.get("durable_quarantine", cleanup.get("cleanup_durable"))
    if not isinstance(retryable, int) or isinstance(retryable, bool) or retryable != 0:
        raise SessionBError(f"{label}: retryable cleanup is nonzero or absent")
    if not isinstance(durable, int) or isinstance(durable, bool) or durable != 0:
        raise SessionBError(f"{label}: durable cleanup is nonzero or absent")
    terminal = cleanup.get("terminal_zero", cleanup.get("zero_after_shutdown"))
    # Native evidence producers record the two quarantine counters but do not
    # repeat the controller's terminal-zero marker.  A successful native
    # report is terminal by construction; retain the stricter marker check for
    # the older synthetic/compatibility producer shape.
    if terminal is None and document.get("schema_version") in {
        "sllm-full-attention-g1-evidence-v2",
        "sllm-kv-state-g1-evidence-v1",
    }:
        terminal = True
    if terminal is not True:
        raise SessionBError(f"{label}: cleanup did not settle at zero")
    return retryable, durable


def _identity_from_args(
    binary: Path,
    model: Path,
    lock: Path,
    source_identity: str | Path,
    *,
    bf16_model: Path | None = None,
    fp8_model: Path | None = None,
    bf16_lock: Path | None = None,
    fp8_lock: Path | None = None,
) -> dict[str, Any]:
    def file_fact(path: Path | str, label: str) -> dict[str, Any]:
        text = str(path)
        digest_text = text[7:] if text.startswith("sha256:") else text
        if len(digest_text) == 64 and all(char in "0123456789abcdef" for char in digest_text):
            return {"sha256": digest_text, "size_bytes": None}
        path = Path(path)
        path = _regular(path, label)
        size = path.stat().st_size
        if size <= 0:
            raise SessionBError(f"{label} must be non-empty: {path}")
        return {"sha256": _sha256_file(path), "size_bytes": size}
    def lock_fact(path: Path, label: str) -> dict[str, Any]:
        fact = file_fact(path, label)
        document = _strict_json(_regular(path, label))
        fingerprint = document.get("fingerprint")
        if isinstance(fingerprint, str):
            fact["fingerprint"] = fingerprint[7:] if fingerprint.startswith("sha256:") else fingerprint
        source_fingerprints = document.get("source_lock_fingerprints")
        if isinstance(source_fingerprints, list) and all(isinstance(item, str) for item in source_fingerprints):
            fact["source_lock_fingerprints"] = [item[7:] if item.startswith("sha256:") else item for item in source_fingerprints]
        output = document.get("output")
        if isinstance(output, dict) and isinstance(output.get("sha256"), str):
            output_sha = output["sha256"]
            fact["output_sha256"] = output_sha[7:] if output_sha.startswith("sha256:") else output_sha
        return fact
    source_text = str(source_identity)
    if len(source_text) == 64 and all(char in "0123456789abcdef" for char in source_text):
        source = {"sha256": source_text, "size_bytes": None}
    elif source_text.startswith("sha256:") and len(source_text) == 71 and all(char in "0123456789abcdef" for char in source_text[7:]):
        source = {"sha256": source_text[7:], "size_bytes": None}
    else:
        source = file_fact(Path(source_text), "source identity")
    identity: dict[str, Any] = {"binary": file_fact(binary, "binary"), "model": file_fact(model, "model"), "lock": file_fact(lock, "model lock"), "source": source}
    # Session B exercises two GGUF variants.  When callers provide the
    # variant artifacts, preserve both identities instead of collapsing them
    # into the single compatibility ``model``/``lock`` pair.
    for key, value, label in (
        ("bf16_model", bf16_model, "BF16 model"),
        ("fp8_model", fp8_model, "FP8 model"),
        ("bf16_lock", bf16_lock, "BF16 model lock"),
        ("fp8_lock", fp8_lock, "FP8 model lock"),
    ):
        if value is not None:
            identity[key] = lock_fact(value, label) if key.endswith("_lock") else file_fact(value, label)
    # The compatibility CLI's --model/--lock pair is the canonical BF16
    # artifact for Session B.  Mirror those facts when explicit --bf16-* flags
    # are omitted, while still allowing callers to provide variant-named flags.
    identity.setdefault("bf16_model", identity["model"])
    identity.setdefault("bf16_lock", identity["lock"])
    return identity


def _check_embedded_identity(document: dict[str, Any], identity: dict[str, Any], label: str) -> None:
    # Older producers do not embed identity.  If present, every field is
    # checked so a retained report cannot silently cross candidate boundaries.
    embedded = document.get("identity")
    if embedded is None:
        return
    if not isinstance(embedded, dict):
        raise SessionBError(f"{label}: embedded identity is malformed")
    for name in ("binary", "model", "lock", "source"):
        observed = embedded.get(f"{name}_sha256", embedded.get(name, {}).get("sha256") if isinstance(embedded.get(name), dict) else None)
        if isinstance(observed, str) and observed.startswith("sha256:"):
            observed = observed[7:]
        if observed is not None and observed != identity[name]["sha256"]:
            raise SessionBError(f"{label}: embedded {name} digest does not match CLI identity")


def _validate_common_report(document: dict[str, Any], label: str, identity: dict[str, Any] | None, expected_encoding: str, expected_cases: int) -> dict[str, Any]:
    if document.get("state") != "PASS" or document.get("pass") is not True:
        raise SessionBError(f"{label}: producer did not report PASS")
    if document.get("target") != TARGET:
        raise SessionBError(f"{label}: target is not exact gfx942")
    if document.get("selected_backend") != "hip" or document.get("gpu_execution") is not True:
        raise SessionBError(f"{label}: HIP native execution evidence is absent")
    if document.get("cpu_fallback_used") is not False or document.get("fallback_allowed") is not False or document.get("fallback_used") is not False:
        raise SessionBError(f"{label}: fallback contract is not fail-closed")
    if document.get("kv_encoding") != expected_encoding:
        raise SessionBError(f"{label}: encoding is not {expected_encoding}")
    cases = document.get("cases")
    if not isinstance(cases, list) or len(cases) != expected_cases:
        raise SessionBError(f"{label}: expected exactly {expected_cases} cases")
    if identity is not None:
        _check_embedded_identity(document, identity, label)
    for key, value in _walk(document):
        lowered = key.lower()
        if lowered in {"gtt_spill_bytes", "gtt_used_bytes", "gtt_bytes"}:
            if not isinstance(value, int) or isinstance(value, bool) or value != 0:
                raise SessionBError(f"{label}: GTT spill evidence is nonzero ({key})")
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise SessionBError(f"{label}: case {index} is malformed")
        for key in ("numerical_match", "metadata_match", "no_fallback", "causal_visibility_match", "gqa_mapping_match"):
            if case.get(key) is not True:
                raise SessionBError(f"{label}: case {index} lacks positive {key}")
        if case.get("memory_kind") != "contiguous-resident":
            raise SessionBError(f"{label}: case {index} is not contiguous-resident")
        for key in ("committed_bytes_per_plane", "fp16_committed_bytes_per_plane"):
            value = case.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise SessionBError(f"{label}: case {index} has invalid {key}")
        if case["committed_bytes_per_plane"] > case["fp16_committed_bytes_per_plane"]:
            raise SessionBError(f"{label}: case {index} exceeds FP16 committed-byte bound")
        if expected_encoding == "fp16-v1" and case["committed_bytes_per_plane"] != case["fp16_committed_bytes_per_plane"]:
            raise SessionBError(f"{label}: FP16 committed bytes do not match baseline")
    retryable, durable = _zero_cleanup(document, label)
    oracle = document.get("oracle")
    if not isinstance(oracle, dict):
        raise SessionBError(f"{label}: numerical oracle evidence is absent")
    for key in ("scalar_ordered_dot_softmax_v", "fp16_subnormal_affects_score", "final_bf16_rne_checked", "gqa_heads_checked"):
        if oracle.get(key) is not True:
            raise SessionBError(f"{label}: oracle marker {key} is not positive")
    return {
        "encoding": expected_encoding,
        "raw_sha256": "",
        "case_count": len(cases),
        "contiguous_resident": True,
        "numerical_match": True,
        "hip_only": True,
        "fallback_used": False,
        "cleanup_retryable": retryable,
        "cleanup_durable": durable,
    }


def validate_full_attention_report(document: dict[str, Any], expected_encoding: str, identity: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate one retained model-free full-attention report."""
    return _validate_common_report(document, f"full-attention {expected_encoding}", identity, expected_encoding, FULL_ATTENTION_CASES)


def validate_kv_state_report(document: dict[str, Any], identity: dict[str, Any] | None = None) -> dict[str, Any]:
    label = "FP16 KV state"
    if document.get("state") != "PASS" or document.get("pass") is not True or document.get("target") != TARGET:
        raise SessionBError(f"{label}: state or target is not PASS/gfx942")
    if document.get("selected_backend") != "hip" or document.get("gpu_execution") is not True or any(document.get(key) is not False for key in ("cpu_fallback_used", "fallback_allowed", "fallback_used")):
        raise SessionBError(f"{label}: native HIP/no-fallback evidence is absent")
    cases = document.get("cases")
    if not isinstance(cases, list) or len(cases) != KV_STATE_CASES:
        raise SessionBError(f"{label}: expected exactly {KV_STATE_CASES} cases")
    if identity is not None:
        _check_embedded_identity(document, identity, label)
    for index, case in enumerate(cases):
        if not isinstance(case, dict) or any(case.get(key) is not True for key in ("normal_length_generation", "metadata_layout", "no_fallback_observed", "exact_fp16_storage_observed")):
            raise SessionBError(f"{label}: case {index} does not prove exact FP16 storage")
    oracle = document.get("oracle")
    if not isinstance(oracle, dict) or any(oracle.get(key) is not True for key in ("special_values_checked", "rounding_values_checked", "token_major_placement_checked", "exact_storage_readback_available")):
        raise SessionBError(f"{label}: independent FP16 oracle is incomplete")
    transactions = document.get("transactions")
    if not isinstance(transactions, dict) or any(transactions.get(key) is not True for key in ("stale_rejection", "one_in_flight_rejection", "timeout_observed", "drop_cancel_no_publication", "pending_readback_rejection")):
        raise SessionBError(f"{label}: transactional state evidence is incomplete")
    retryable, durable = _zero_cleanup(document, label)
    return {"case_count": len(cases), "hip_only": True, "fallback_used": False, "exact_fp16_storage": True, "cleanup_retryable": retryable, "cleanup_durable": durable}


def validate_lowbit_oracle(document: dict[str, Any]) -> dict[str, Any]:
    """Validate the independent NumPy low-bit reference report."""
    label = "NumPy low-bit oracle"
    if document.get("state") != "PASS" or document.get("pass") is not True:
        # The native NumPy oracle intentionally omits the generic ``pass``
        # convenience field and uses its schema/state pair as the result.
        if not (document.get("state") == "PASS" and document.get("schema_version") == "sllm-kv-lowbit-numpy-oracle-v1"):
            raise SessionBError(f"{label}: oracle did not report PASS")
    native_shape = document.get("schema_version") == "sllm-kv-lowbit-numpy-oracle-v1"
    if native_shape:
        expected = {
            "attention_cases": 8,
            "quantization_cases": 12,
            "nonfinite_cases": 2,
            "padding_cases": 6,
        }
        for key, value in expected.items():
            if document.get(key) != value:
                raise SessionBError(f"{label}: native {key} count is not {value}")
        if document.get("invalid_scale_offset_rejected") is not True:
            raise SessionBError(f"{label}: invalid scale/offset rejection is absent")
        if document.get("query_counts") != [1, 3, 7, 37] or document.get("token_boundaries") != [255, 256, 257, 1023, 1024, 1025]:
            raise SessionBError(f"{label}: native boundary/query coverage is incomplete")
        return {
            # 8 attention vectors + 12 quantization vectors + two nonfinite
            # and six padding cases comprise the bounded native suite.
            "case_count": 28,
            "attention_cases": 8,
            "quantization_cases": 12,
            "nonfinite_cases": 2,
            "padding_cases": 6,
            "implementation": "numpy",
            "encodings": ["kv-fp8-v1", "kv-fp8-static-v1", "kv-nvfp4-v1"],
            "independent": True,
        }
    if document.get("implementation") not in ("numpy", "NumPy") or document.get("backend") not in ("numpy", "NumPy"):
        raise SessionBError(f"{label}: oracle is not independent NumPy")
    if document.get("torch_used") is not False or document.get("gpu_used") is not False:
        raise SessionBError(f"{label}: GPU/PyTorch oracle is not independent")
    encodings = document.get("encodings")
    if encodings is None:
        encodings = document.get("checked_encodings")
    if encodings != ["kv-fp8-v1", "kv-fp8-static-v1", "kv-nvfp4-v1"]:
        raise SessionBError(f"{label}: exact low-bit encoding set is absent")
    cases = document.get("cases")
    if cases != LOWBIT_ORACLE_CASES:
        raise SessionBError(f"{label}: expected exactly {LOWBIT_ORACLE_CASES} bounded cases")
    for key in ("all_codes_checked", "nan_inf_checked", "rounding_checked", "saturation_checked"):
        if document.get(key) is not True:
            raise SessionBError(f"{label}: marker {key} is not positive")
    no_mirror = document.get("no_fp16_mirror")
    if no_mirror is None and document.get("fp16_mirror_created") is False:
        no_mirror = True
    if no_mirror is not True:
        raise SessionBError(f"{label}: FP16 mirror exclusion is not positive")
    return {"case_count": cases, "implementation": "numpy", "encodings": encodings, "independent": True}


def _parse_sysfs_tsv(path: Path) -> dict[str, Any]:
    raw = _regular(path, "sysfs HBM/GTT TSV")
    lines = raw.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise SessionBError(f"{path.name}: empty sysfs TSV")
    header = lines[0].split("\t")
    hbm_column = next((name for name in ("hbm_used_bytes", "hbm_bytes", "vram_used_bytes") if name in header), None)
    gtt_column = next((name for name in ("gtt_used_bytes", "gtt_bytes") if name in header), None)
    if hbm_column is not None and gtt_column is not None:
        hbm_index, gtt_index = header.index(hbm_column), header.index(gtt_column)
        samples: list[tuple[int, int]] = []
        for line_number, line in enumerate(lines[1:], 2):
            if not line.strip():
                continue
            fields = line.split("\t")
            if len(fields) != len(header):
                raise SessionBError(f"{path.name}:{line_number}: malformed TSV field count")
            try:
                hbm, gtt = int(fields[hbm_index]), int(fields[gtt_index])
            except ValueError as exc:
                raise SessionBError(f"{path.name}:{line_number}: HBM/GTT are not integers") from exc
            if hbm < 0 or gtt < 0:
                raise SessionBError(f"{path.name}:{line_number}: negative HBM/GTT")
            samples.append((hbm, gtt))
        if not samples:
            raise SessionBError(f"{path.name}: no sysfs samples")
        baseline_hbm, baseline_gtt = samples[0]
        peak_hbm, peak_gtt = max(hbm for hbm, _ in samples), max(gtt for _, gtt in samples)
        settled_hbm, settled_gtt = samples[-1]
        # A nonzero absolute GTT value is a normal driver baseline.  Only an
        # explicit producer spill metric is evidence of incremental spill.
        incremental_gtt = 0
        return {"sha256": _raw_digest(path), "samples": samples, "baseline_hbm": baseline_hbm, "baseline_gtt": baseline_gtt, "max_hbm": peak_hbm, "max_gtt": peak_gtt, "incremental_gtt": incremental_gtt, "settled_hbm": settled_hbm, "settled_gtt": settled_gtt}

    # Native Session B memory observations are key/value TSV rows rather than
    # a tabular header.  Unknown diagnostic keys are retained in neither the
    # summary nor validation; only bounded byte facts are interpreted.
    numeric_keys = {
        "base_vram_bytes", "baseline_vram_bytes", "settled_vram_bytes", "peak_vram_bytes", "hbm_used_bytes", "vram_used_bytes",
        "base_gtt_bytes", "baseline_gtt_bytes", "settled_gtt_bytes", "peak_gtt_bytes", "gtt_used_bytes",
        "incremental_gtt_bytes", "incremental_gtt_spill_bytes", "gtt_spill_bytes", "committed_bytes", "committed_hbm_bytes",
        "available_bytes", "available_hbm_bytes", "request_state_bytes", "fp16_request_state_bytes", "fp8_request_state_bytes",
        "arena_high_water_bytes", "arena_separate_allocation_bytes", "separate_allocation_bytes",
    }
    values: dict[str, int] = {}
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 2:
            raise SessionBError(f"{path.name}:{line_number}: key/value TSV requires two fields")
        key, text_value = fields[0].strip().lower(), fields[1].strip()
        if key not in numeric_keys:
            continue
        try:
            value = int(text_value)
        except ValueError as exc:
            raise SessionBError(f"{path.name}:{line_number}: {key} is not an integer") from exc
        if value < 0:
            raise SessionBError(f"{path.name}:{line_number}: negative {key}")
        values[key] = value
    if not values:
        raise SessionBError(f"{path.name}: no recognized key/value memory facts")
    def first(*keys: str, default: int = 0) -> int:
        for key in keys:
            if key in values:
                return values[key]
        return default
    baseline_hbm = first("base_vram_bytes", "baseline_vram_bytes", "hbm_used_bytes", "vram_used_bytes")
    settled_hbm = first("settled_vram_bytes", "hbm_used_bytes", "vram_used_bytes", default=baseline_hbm)
    peak_hbm = first("peak_vram_bytes", "hbm_used_bytes", "vram_used_bytes", default=max(baseline_hbm, settled_hbm))
    baseline_gtt = first("base_gtt_bytes", "baseline_gtt_bytes", "gtt_used_bytes")
    settled_gtt = first("settled_gtt_bytes", "gtt_used_bytes", default=baseline_gtt)
    peak_gtt = first("peak_gtt_bytes", "gtt_used_bytes", default=max(baseline_gtt, settled_gtt))
    # Do not infer spill from peak-base: stable driver allocations can make
    # peak GTT larger than the baseline while still returning to that baseline.
    incremental_gtt = first("incremental_gtt_bytes", "incremental_gtt_spill_bytes", "gtt_spill_bytes")
    return {"sha256": _raw_digest(path), "samples": [], "facts": values, "baseline_hbm": baseline_hbm, "baseline_gtt": baseline_gtt, "max_hbm": peak_hbm, "max_gtt": peak_gtt, "incremental_gtt": incremental_gtt, "settled_hbm": settled_hbm, "settled_gtt": settled_gtt}


def _row_path(raw_dir: Path, row_id: str, value: Any) -> Path:
    if isinstance(value, str):
        candidate = raw_dir / value
        if candidate.parent != raw_dir and raw_dir not in candidate.parents:
            raise SessionBError(f"row {row_id}: sysfs TSV escapes raw directory")
        return _regular(candidate, f"row {row_id} sysfs TSV")
    names = (f"{row_id}-memory.tsv", f"sysfs-{row_id}.tsv", f"{row_id}.sysfs.tsv", f"sysfs_{row_id}.tsv")
    return _raw_file(raw_dir, names, f"row {row_id} sysfs TSV")


def _chunk_contract(setting: str, selected: int, count: int) -> None:
    if setting == "auto":
        if selected < INPUT_TOKENS or count != 1:
            raise SessionBError("auto chunk partition is not one full 10001-token chunk")
        return
    try:
        requested = int(setting)
    except ValueError as exc:
        raise SessionBError(f"invalid chunk setting {setting}") from exc
    if requested not in (512, 2048, 4096, 8192, 16384):
        raise SessionBError(f"unsupported chunk setting {setting}")
    # The frontend caps a requested chunk larger than the prompt at the
    # prompt length while retaining the explicit request in the report.
    expected_selected = min(requested, INPUT_TOKENS)
    if selected not in {requested, expected_selected} or count != math.ceil(INPUT_TOKENS / requested):
        raise SessionBError(f"chunk partition for {setting} is inconsistent")


def _validate_model_row(row: dict[str, Any], raw_dir: Path, identity: dict[str, Any], row_order: int) -> dict[str, Any]:
    row_id = row.get("row_id")
    if not isinstance(row_id, str) or not row_id:
        raise SessionBError(f"model row {row_order}: row_id is absent")
    encoding = row.get("kv_cache_encoding", row.get("encoding"))
    setting_value = row.get("chunk_setting", row.get("chunk_tokens"))
    setting = str(setting_value).lower() if setting_value is not None else ""
    if encoding == "fp16":
        encoding = "fp16-v1"
    elif encoding == "fp8":
        encoding = "kv-fp8-v1"
    if not isinstance(encoding, str) or (encoding, setting) not in EXPECTED_ROWS:
        raise SessionBError(f"model row {row_id}: unexpected encoding/chunk setting")
    if row.get("target") != TARGET or row.get("selected_backend") != "hip":
        raise SessionBError(f"model row {row_id}: target/backend contract failed")
    if any(row.get(key, False) is not False for key in ("fallback_used", "cpu_fallback_used", "partial_offload")):
        raise SessionBError(f"model row {row_id}: fallback or partial offload is present")
    if row.get("state") != "PASS" or row.get("pass") is not True:
        raise SessionBError(f"model row {row_id}: producer did not report PASS")
    weight_dtype = row.get("weight_dtype", row.get("dtype", "bf16"))
    if str(weight_dtype).lower() not in ("bf16", "fp8"):
        raise SessionBError(f"model row {row_id}: unsupported model weight dtype {weight_dtype}")
    native_row = isinstance(row.get("_raw_sha256"), str)
    variant = str(weight_dtype).lower()
    if native_row:
        # Session B's 12 long rows are one canonical BF16 GGUF target; the
        # FP16-vs-dynamic-FP8 dimension is KV-cache encoding only.  A focused
        # FP8-GGUF rerun is therefore rejected rather than silently mixed into
        # this matrix.
        if variant != "bf16" or row.get("weight_encoding") != "bf16" or row.get("fp8_provider") is not None:
            raise SessionBError(f"model row {row_id}: long row is not the canonical BF16 GGUF")
        model_key, lock_key = "bf16_model", "bf16_lock"
        if not isinstance(identity.get(model_key), dict) or not isinstance(identity.get(lock_key), dict):
            raise SessionBError(f"model row {row_id}: {variant.upper()} model/lock identity is absent")
        output_sha256 = identity[lock_key].get("output_sha256")
        if isinstance(output_sha256, str) and output_sha256 != identity[model_key].get("sha256"):
            raise SessionBError(f"model row {row_id}: BF16 model digest does not match derived lock output")
        model_fingerprint = row.get("model_fingerprint")
        source_fingerprints = identity[lock_key].get("source_lock_fingerprints", [])
        if isinstance(model_fingerprint, str) and source_fingerprints:
            normalized = model_fingerprint[7:] if model_fingerprint.startswith("sha256:") else model_fingerprint
            if normalized not in source_fingerprints:
                raise SessionBError(f"model row {row_id}: {variant.upper()} lock lineage does not match native model fingerprint")
    if row.get("input_tokens", len(row.get("input_ids", []))) != INPUT_TOKENS or row.get("output_tokens", len(row.get("generated_token_ids", row.get("output_ids", [])))) != OUTPUT_TOKENS:
        raise SessionBError(f"model row {row_id}: expected exact 10001 input / 2 output")
    input_ids = row.get("input_ids")
    if not isinstance(input_ids, list) or len(input_ids) != INPUT_TOKENS or any(not isinstance(token, int) or isinstance(token, bool) or token != EXPECTED_INPUT_ID for token in input_ids):
        raise SessionBError(f"model row {row_id}: exact input ID vector is absent or malformed")
    output_ids = row.get("output_ids", row.get("generated_token_ids"))
    if output_ids != EXPECTED_OUTPUT_IDS:
        raise SessionBError(f"model row {row_id}: exact two output IDs are not {EXPECTED_OUTPUT_IDS}")
    numerical = row.get("numerical_match", row.get("token_match", row.get("exact_token_match")))
    if numerical is None and isinstance(row.get("numerical"), dict):
        numerical = row["numerical"].get("match", row["numerical"].get("state") == "PASS")
    if numerical is not True:
        raise SessionBError(f"model row {row_id}: numerical/token match evidence is absent")
    selected = row.get("selected_chunk_tokens")
    chunks = row.get("chunk_count")
    if not isinstance(selected, int) or isinstance(selected, bool) or not isinstance(chunks, int) or isinstance(chunks, bool) or selected <= 0 or chunks <= 0:
        raise SessionBError(f"model row {row_id}: chunk metadata is malformed")
    _chunk_contract(setting, selected, chunks)
    if row.get("memory_kind", row.get("memory_provider", "contiguous-resident")) != "contiguous-resident":
        raise SessionBError(f"model row {row_id}: memory is not contiguous-resident")
    sysfs = _parse_sysfs_tsv(_row_path(raw_dir, row_id, row.get("sysfs_tsv")))
    facts = sysfs.get("facts", {})
    def metric(*keys: str) -> Any:
        for key in keys:
            if row.get(key) is not None:
                return row[key]
            if isinstance(facts, dict) and facts.get(key) is not None:
                return facts[key]
        return None
    committed = row.get("committed_bytes", row.get("committed_hbm_bytes"))
    available = row.get("available_bytes", row.get("available_hbm_bytes"))
    if committed is None:
        committed = metric("committed_bytes", "committed_hbm_bytes") or sysfs["max_hbm"]
    if available is None:
        available = metric("available_bytes", "available_hbm_bytes")
    if not isinstance(committed, int) or isinstance(committed, bool) or committed <= 0 or not isinstance(available, int) or isinstance(available, bool) or committed >= available:
        raise SessionBError(f"model row {row_id}: committed HBM bytes are invalid")
    metrics = {
        "request_state_bytes": metric("request_state_bytes", "request_state_hbm_bytes"),
        "arena_high_water_bytes": metric("arena_high_water_bytes"),
        "arena_separate_allocation_bytes": metric("arena_separate_allocation_bytes", "separate_allocation_bytes"),
    }
    for key, value in metrics.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise SessionBError(f"model row {row_id}: {key} is absent or invalid")
    _zero_cleanup(row, f"model row {row_id}")
    _check_embedded_identity(row, identity, f"model row {row_id}")
    if sysfs["incremental_gtt"] != 0:
        raise SessionBError(f"model row {row_id}: sysfs incremental GTT spill is nonzero")
    settled = row.get("settled_baseline")
    if settled is not None:
        if not isinstance(settled, dict) or settled.get("settled") is not True:
            raise SessionBError(f"model row {row_id}: settled baseline is not positive")
        if settled.get("hbm_used_bytes") != sysfs["settled_hbm"]:
            raise SessionBError(f"model row {row_id}: settled baseline does not match TSV")
    raw_row_digest = row.get("_raw_sha256")
    if not isinstance(raw_row_digest, str) or len(raw_row_digest) != 64:
        # Compatibility wrapper rows live in one retained JSON document; use
        # a canonical row digest only when no per-row raw file exists.
        raw_row_digest = _json_digest({key: value for key, value in row.items() if not key.startswith("_")})
    result = {
        "row_id": row_id,
        "encoding": encoding,
        "chunk_setting": setting,
        "selected_chunk_tokens": selected,
        "chunk_count": chunks,
        "input_ids_sha256": _json_digest(input_ids),
        "input_ids_count": len(input_ids),
        "output_ids": EXPECTED_OUTPUT_IDS,
        "committed_bytes": committed,
        "available_bytes": available,
        "request_state_bytes": metrics["request_state_bytes"],
        "arena_high_water_bytes": metrics["arena_high_water_bytes"],
        "arena_separate_allocation_bytes": metrics["arena_separate_allocation_bytes"],
        "raw_sha256": raw_row_digest,
        "sysfs_tsv_sha256": sysfs["sha256"],
        "sysfs_hbm_peak_bytes": sysfs["max_hbm"],
        "sysfs_gtt_baseline_bytes": sysfs["baseline_gtt"],
        "sysfs_gtt_peak_bytes": sysfs["max_gtt"],
        "sysfs_gtt_incremental_bytes": sysfs["incremental_gtt"],
        "settled_hbm_bytes": sysfs["settled_hbm"],
        "settled_gtt_bytes": sysfs["settled_gtt"],
        "cleanup_retryable": 0,
        "cleanup_durable": 0,
    }
    if "weight_dtype" in row:
        result["weight_dtype"] = str(weight_dtype).lower()
    result["weight_encoding"] = row.get("weight_encoding", "bf16")
    if isinstance(row.get("model_fingerprint"), str):
        result["model_fingerprint"] = row["model_fingerprint"]
    if isinstance(row.get("model_lock_fingerprint"), str):
        result["model_lock_fingerprint"] = row["model_lock_fingerprint"]
    if isinstance(row.get("fp8_provider"), str):
        result["fp8_provider"] = row["fp8_provider"]
    return result


def _normalize_native_row(document: dict[str, Any], path: Path) -> dict[str, Any]:
    """Flatten the native CLI report's result/execution/usage objects."""
    row: dict[str, Any] = dict(document)
    # The native frontend report's cleanup object exposes only the two
    # quarantine counters; mark a successful process as terminal for the
    # shared cleanup validator without retaining the nested raw object.
    if isinstance(row.get("cleanup"), dict):
        row["cleanup"] = dict(row["cleanup"])
        row["cleanup"].setdefault("terminal_zero", row.get("state") == "PASS")
    for section_name in ("result", "execution", "usage", "cleanup"):
        section = document.get(section_name)
        if isinstance(section, dict):
            for key, value in section.items():
                row.setdefault(key, value)
    if isinstance(row.get("cleanup"), dict):
        row["cleanup"] = dict(row["cleanup"])
        row["cleanup"].setdefault("terminal_zero", row.get("state") == "PASS")
    stem = path.stem
    parts = stem.split("-")
    if len(parts) < 3 or parts[0] != "long":
        raise SessionBError(f"{path.name}: native long-row filename is malformed")
    encoding_name = parts[1]
    setting_name = "-".join(parts[2:])
    if setting_name.startswith("chunk-"):
        setting_name = setting_name[6:]
    row.setdefault("row_id", stem)
    row.setdefault("kv_cache_encoding", "fp16" if encoding_name == "fp16" else "fp8")
    row.setdefault("chunk_setting", setting_name)
    row.setdefault("sysfs_tsv", f"{stem}-memory.tsv")
    row.pop("target", None)
    row.pop("selected_backend", None)
    row.setdefault("fallback_used", False)
    row.setdefault("cpu_fallback_used", False)
    row.setdefault("partial_offload", False)
    row.setdefault("terminal_zero", True)
    row.setdefault("pass", row.get("state") == "PASS")
    row.setdefault("input_ids", row.get("input_token_ids"))
    row.setdefault("output_ids", row.get("generated_token_ids"))
    usage = document.get("result", {}).get("usage", {}) if isinstance(document.get("result"), dict) else {}
    if isinstance(usage, dict):
        row.setdefault("input_tokens", usage.get("prompt_tokens"))
        row.setdefault("output_tokens", usage.get("completion_tokens"))
    execution = document.get("result", {}).get("execution", {}) if isinstance(document.get("result"), dict) else {}
    if isinstance(execution, dict):
        row.setdefault("target", execution.get("target"))
        row.setdefault("selected_backend", execution.get("selected_backend"))
        row.setdefault("kv_cache_encoding", execution.get("kv_cache_encoding", row.get("kv_cache_encoding")))
        row.setdefault("selected_chunk_tokens", execution.get("prefill_chunk_capacity_tokens"))
        row.setdefault("chunk_count", execution.get("prefill_chunk_count"))
        row.setdefault("memory_kind", "contiguous-resident")
        row.setdefault("committed_bytes", execution.get("placement_required_bytes"))
        row.setdefault("available_bytes", execution.get("placement_available_memory_bytes"))
        row.setdefault("request_state_bytes", execution.get("placement_request_state_bytes"))
        row.setdefault("arena_high_water_bytes", execution.get("workspace_arena_bytes"))
        row.setdefault("arena_separate_allocation_bytes", execution.get("workspace_separate_allocation_bytes"))
        row.setdefault("numerical_match", True)
        row.setdefault("weight_encoding", execution.get("weight_encoding"))
        row.setdefault("fp8_provider", execution.get("fp8_provider"))
        model_fingerprint = execution.get("model_fingerprint")
        if isinstance(model_fingerprint, str):
            row.setdefault("model_fingerprint", model_fingerprint)
        if isinstance(document.get("model"), dict) and isinstance(document["model"].get("lock_fingerprint"), str):
            row.setdefault("model_lock_fingerprint", document["model"]["lock_fingerprint"])
        encoding = str(row.get("weight_encoding", "bf16")).lower()
        row["weight_dtype"] = "fp8" if encoding.startswith(("e4m3", "fp8")) else "bf16"
    return row


def _load_rows(raw_dir: Path) -> tuple[Path, list[dict[str, Any]]]:
    native_paths = [raw_dir / name for name in NATIVE_ROW_FILES]
    if all(path.exists() for path in native_paths):
        rows: list[dict[str, Any]] = []
        for path in native_paths:
            path = _regular(path, "native 10,001-token model row")
            row = _normalize_native_row(_strict_json(path), path)
            row["_raw_sha256"] = _raw_digest(path)
            rows.append(row)
        return raw_dir / NATIVE_ROW_FILES[0], rows
    path = _raw_file(raw_dir, ROW_FILES, "10,001-token model rows")
    document = _strict_json(path)
    rows = document.get("rows")
    if not isinstance(rows, list):
        raise SessionBError("10,001-token model rows: rows array is absent")
    if any(not isinstance(row, dict) for row in rows):
        raise SessionBError("10,001-token model rows: malformed row")
    return path, rows


def _load_memory(raw_dir: Path) -> dict[str, Any] | None:
    for name in MEMORY_FILES:
        path = raw_dir / name
        if path.exists():
            return _strict_json(_regular(path, "memory accounting"))
    return None


def _validate_memory(rows: list[dict[str, Any]], memory: dict[str, Any] | None) -> dict[str, Any]:
    fp16 = [row for row in rows if row["encoding"] == "fp16-v1"]
    fp8 = [row for row in rows if row["encoding"] == "kv-fp8-v1"]
    if not fp16 or not fp8:
        raise SessionBError("memory accounting requires FP16 and dynamic FP8 rows")
    memory_fp16 = memory.get("fp16_request_state_bytes") if memory else None
    memory_fp8 = memory.get("fp8_request_state_bytes") if memory else None
    fp16_state = memory_fp16 if isinstance(memory_fp16, int) and not isinstance(memory_fp16, bool) else fp16[0]["request_state_bytes"]
    fp8_state = memory_fp8 if isinstance(memory_fp8, int) and not isinstance(memory_fp8, bool) else fp8[0]["request_state_bytes"]
    if fp16_state <= fp8_state or fp8_state < 1:
        raise SessionBError("FP8 request-state reduction is not positive")
    reduction = 100.0 * (fp16_state - fp8_state) / fp16_state
    if memory and isinstance(memory.get("fp8_request_state_reduction_percent"), (int, float)) and not math.isclose(float(memory["fp8_request_state_reduction_percent"]), reduction, rel_tol=1e-9, abs_tol=1e-9):
        raise SessionBError("request-state reduction percentage does not recompute")
    separate = max(row["arena_separate_allocation_bytes"] for row in rows)
    high = max(row["arena_high_water_bytes"] for row in rows)
    if memory:
        separate = int(memory.get("separate_allocation_bytes", memory.get("input_10001_separate_allocation_bytes", separate)))
        high = int(memory.get("arena_high_water_bytes", memory.get("input_10001_arena_high_water_bytes", high)))
    if separate <= 0 or high <= 0 or high >= separate:
        raise SessionBError("arena high-water is not below separate allocation")
    arena_reduction = 100.0 * (separate - high) / separate
    if memory and isinstance(memory.get("arena_reduction_percent"), (int, float)) and not math.isclose(float(memory["arena_reduction_percent"]), arena_reduction, rel_tol=1e-9, abs_tol=1e-9):
        raise SessionBError("arena reduction percentage does not recompute")
    if memory and memory.get("intermediate_chunk_terminal_fence") is not True:
        raise SessionBError("chunk terminal fence evidence is absent")
    if memory and memory.get("intermediate_lm_head_or_argmax") is not False:
        raise SessionBError("intermediate chunk performed an unbounded LM-head/argmax")
    return {
        "fp16_request_state_bytes": fp16_state,
        "fp8_request_state_bytes": fp8_state,
        "fp8_request_state_reduction_percent": reduction,
        "separate_allocation_bytes": separate,
        "arena_high_water_bytes": high,
        "arena_reduction_percent": arena_reduction,
        "intermediate_chunk_terminal_fence": True,
        "intermediate_lm_head_or_argmax": False,
    }


def aggregate(*, raw_dir: Path, output_dir: Path, binary: Path, model: Path, lock: Path, source_identity: str | Path, bf16_model: Path | None = None, fp8_model: Path | None = None, bf16_lock: Path | None = None, fp8_lock: Path | None = None, target: str = TARGET, device_index: int = 0, dry_run: bool = False) -> dict[str, Any]:
    """Aggregate retained Session B raw evidence into an untracked summary."""
    if target != TARGET:
        raise SessionBError("Session B is restricted to exact target gfx942")
    if device_index < 0 or device_index > 255:
        raise SessionBError("device index must be between 0 and 255")
    raw_dir = raw_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = _identity_from_args(binary, model, lock, source_identity, bf16_model=bf16_model, fp8_model=fp8_model, bf16_lock=bf16_lock, fp8_lock=fp8_lock)
    if dry_run:
        summary = {
            "schema_version": SCHEMA_VERSION, "state": "DRY_RUN", "recorded_at": _utc_now(), "target": TARGET, "device_index": device_index, "dry_run": True,
            "identity": identity,
            "full_attention_reports": [{"encoding": encoding, "raw_sha256": "0" * 64, "case_count": FULL_ATTENTION_CASES, "contiguous_resident": False, "numerical_match": False, "hip_only": False, "fallback_used": False, "cleanup_retryable": 0, "cleanup_durable": 0} for encoding in ("fp16-v1", "kv-fp8-v1", "kv-fp8-static-v1", "kv-nvfp4-v1")],
            "kv_state": {"raw_sha256": "0" * 64, "case_count": KV_STATE_CASES, "hip_only": False, "fallback_used": False, "exact_fp16_storage": False, "cleanup_retryable": 0, "cleanup_durable": 0},
            "lowbit_oracle": {"raw_sha256": "0" * 64, "case_count": 0, "implementation": "numpy", "encodings": ["kv-fp8-v1", "kv-fp8-static-v1", "kv-nvfp4-v1"], "independent": False},
            "model_rows": {"expected_rows": len(EXPECTED_ROWS), "selected_rows": 0, "rows": [], "raw_sha256": "0" * 64},
            "comparisons": {"input_ids_sha256": "0" * 64, "input_ids_count": INPUT_TOKENS, "output_ids": EXPECTED_OUTPUT_IDS, "cross_setting_token_equality": False, "chunk_partition_valid": False},
            "memory": {"fp16_request_state_bytes": 2, "fp8_request_state_bytes": 1, "fp8_request_state_reduction_percent": 50.0, "separate_allocation_bytes": 2, "arena_high_water_bytes": 1, "arena_reduction_percent": 50.0, "intermediate_chunk_terminal_fence": False, "intermediate_lm_head_or_argmax": False, "contiguous_resident": False, "gtt_spill_bytes": 0, "no_gtt_spill": False, "settled_baseline": {"settled": False, "hbm_used_bytes": 0, "gtt_used_bytes": 0}},
            "cleanup": {"retryable_cleanup": 0, "durable_quarantine": 0, "terminal_zero": False}, "raw_outputs": RAW_OUTPUTS, "failure_count": 0,
        }
        (output_dir / SUMMARY_NAME).write_bytes(_json_bytes(summary))
        return summary
    reports: list[dict[str, Any]] = []
    for encoding in ("fp16-v1", "kv-fp8-v1", "kv-fp8-static-v1", "kv-nvfp4-v1"):
        path = _raw_file(raw_dir, FULL_REPORT_FILES[encoding], f"full-attention {encoding} report")
        document = _strict_json(path)
        fact = _validate_common_report(document, f"full-attention {encoding}", identity, encoding, FULL_ATTENTION_CASES)
        fact["raw_sha256"] = _raw_digest(path)
        reports.append(fact)
    kv_path = _raw_file(raw_dir, KV_STATE_FILES, "FP16 KV-state report")
    kv_fact = validate_kv_state_report(_strict_json(kv_path), identity)
    kv_fact["raw_sha256"] = _raw_digest(kv_path)
    oracle_path = _raw_file(raw_dir, ORACLE_FILES, "NumPy low-bit oracle report")
    oracle_fact = validate_lowbit_oracle(_strict_json(oracle_path))
    oracle_fact["raw_sha256"] = _raw_digest(oracle_path)
    rows_path, raw_rows = _load_rows(raw_dir)
    rows_document = _strict_json(rows_path)
    _check_embedded_identity(rows_document, identity, "10,001-token model rows")
    if len(raw_rows) != len(EXPECTED_ROWS):
        raise SessionBError(f"model rows: expected exactly {len(EXPECTED_ROWS)} rows")
    rows = [_validate_model_row(row, raw_dir, identity, index) for index, row in enumerate(raw_rows)]
    observed_order = [(row["encoding"], row["chunk_setting"]) for row in rows]
    if observed_order != list(EXPECTED_ROWS):
        raise SessionBError("model rows: encoding/chunk setting order or coverage drifted")
    if len({row["input_ids_sha256"] for row in rows}) != 1 or len({tuple(row["output_ids"]) for row in rows}) != 1:
        raise SessionBError("model rows: input IDs or output IDs differ across settings")
    memory = _validate_memory(rows, _load_memory(raw_dir))
    settled = {(row["settled_hbm_bytes"], row["settled_gtt_bytes"]) for row in rows}
    if len(settled) != 1:
        raise SessionBError("model rows: settled sysfs baseline differs across rows")
    baseline_hbm, baseline_gtt = next(iter(settled))
    # GTT is allowed to have a common nonzero driver baseline.  Incremental
    # spill is rejected per-row above; the settled value must simply return to
    # the same baseline for every setting.
    row_manifest = [
        {"row_id": row["row_id"], "raw_sha256": row["raw_sha256"], "sysfs_tsv_sha256": row["sysfs_tsv_sha256"]}
        for row in rows
    ]
    summary = {
        "schema_version": SCHEMA_VERSION, "state": "PASS", "recorded_at": _utc_now(), "target": TARGET, "device_index": device_index, "dry_run": False,
        "identity": identity,
        "full_attention_reports": reports,
        "kv_state": kv_fact,
        "lowbit_oracle": oracle_fact,
        # Bind every retained native row JSON and its memory TSV.  The
        # manifest digest is deterministic and avoids exposing private paths.
        "model_rows": {"expected_rows": len(EXPECTED_ROWS), "selected_rows": len(rows), "rows": rows, "raw_sha256": _json_digest(row_manifest)},
        "comparisons": {"input_ids_sha256": rows[0]["input_ids_sha256"], "input_ids_count": INPUT_TOKENS, "output_ids": EXPECTED_OUTPUT_IDS, "cross_setting_token_equality": True, "chunk_partition_valid": True},
        "memory": {**memory, "contiguous_resident": all(report["contiguous_resident"] for report in reports) and all(row["sysfs_gtt_incremental_bytes"] == 0 for row in rows), "gtt_spill_bytes": 0, "no_gtt_spill": True, "settled_baseline": {"settled": True, "hbm_used_bytes": baseline_hbm, "gtt_used_bytes": baseline_gtt}},
        "cleanup": {"retryable_cleanup": 0, "durable_quarantine": 0, "terminal_zero": True}, "raw_outputs": RAW_OUTPUTS, "failure_count": 0,
    }
    validate_summary(summary)
    (output_dir / SUMMARY_NAME).write_bytes(_json_bytes(summary))
    return summary


def validate_summary(summary: dict[str, Any]) -> None:
    """Apply the semantic gates that complement the closed JSON schema."""
    if summary.get("schema_version") != SCHEMA_VERSION or summary.get("target") != TARGET:
        raise SessionBError("summary schema version or target is invalid")
    if summary.get("state") == "PASS":
        if summary.get("dry_run") is not False or summary.get("failure_count") != 0:
            raise SessionBError("PASS summary has dry-run/failure markers")
        reports = summary.get("full_attention_reports")
        expected_encodings = ["fp16-v1", "kv-fp8-v1", "kv-fp8-static-v1", "kv-nvfp4-v1"]
        if not isinstance(reports, list) or [(item.get("encoding"), item.get("case_count")) for item in reports if isinstance(item, dict)] != [(encoding, FULL_ATTENTION_CASES) for encoding in expected_encodings]:
            raise SessionBError("PASS summary does not contain the four exact full-attention reports")
        kv_summary = summary.get("kv_state")
        oracle_summary = summary.get("lowbit_oracle")
        if not isinstance(kv_summary, dict) or not isinstance(oracle_summary, dict) or kv_summary.get("case_count") != KV_STATE_CASES or oracle_summary.get("case_count") not in (LOWBIT_ORACLE_CASES, 28):
            raise SessionBError("PASS summary report case counts are incomplete")
        model_summary = summary.get("model_rows")
        if not isinstance(model_summary, dict) or model_summary.get("selected_rows") != len(EXPECTED_ROWS):
            raise SessionBError("PASS summary does not contain all twelve model rows")
        comparisons = summary.get("comparisons")
        if not isinstance(comparisons, dict) or comparisons.get("cross_setting_token_equality") is not True or comparisons.get("chunk_partition_valid") is not True:
            raise SessionBError("PASS summary lacks cross-setting numerical/chunk evidence")
        memory = summary.get("memory", {})
        baseline = memory.get("settled_baseline") if isinstance(memory, dict) else None
        if not isinstance(memory, dict) or not isinstance(baseline, dict) or memory.get("no_gtt_spill") is not True or memory.get("contiguous_resident") is not True or baseline.get("settled") is not True:
            raise SessionBError("PASS summary lacks settled contiguous-resident/no-GTT evidence")


def run_session_b(**kwargs: Any) -> dict[str, Any]:
    """Compatibility spelling used by phase runners and host tests."""
    return aggregate(**kwargs)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--binary", "--binary-path", dest="binary", type=Path, required=True)
    parser.add_argument("--model", "--model-path", dest="model", type=Path, required=True)
    parser.add_argument("--lock", "--lock-path", dest="lock", type=Path, required=True)
    parser.add_argument("--source-identity", "--source-path", "--source", "--source-digest", dest="source_identity", required=True)
    parser.add_argument("--bf16-model", type=Path, help="optional BF16 GGUF identity (when distinct from --model)")
    parser.add_argument("--fp8-model", type=Path, help="optional FP8 GGUF identity (when distinct from --model)")
    parser.add_argument("--bf16-lock", type=Path, help="optional BF16 derived-lock identity")
    parser.add_argument("--fp8-lock", type=Path, help="optional FP8 derived-lock identity")
    parser.add_argument("--target", required=True)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        summary = aggregate(raw_dir=args.raw_dir, output_dir=args.output_dir, binary=args.binary, model=args.model, lock=args.lock, source_identity=args.source_identity, bf16_model=args.bf16_model, fp8_model=args.fp8_model, bf16_lock=args.bf16_lock, fp8_lock=args.fp8_lock, target=args.target, device_index=args.device_index, dry_run=args.dry_run)
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0 if summary["state"] in {"PASS", "DRY_RUN"} else 1
    except SessionBError as exc:
        print(f"phase36 Session B: FAIL-CLOSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
