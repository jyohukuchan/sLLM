#!/usr/bin/env python3
"""Validate SQ8 M=1 serving captures against the checked-in source oracle."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import re
import stat
import struct
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "ullm.sq8.serving_smoke.v2"
RESULT_SCHEMA_VERSION = "ullm.sq8.serving_runtime_oracle_validation.v1"
PROMPT_LENGTHS = (1, 8, 32, 128)
HIDDEN_SIZE = 5_120
VOCAB_SIZE = 151_936
TOP_K = 10
MAX_RELATIVE_L2 = 0.20
MIN_COSINE_SIMILARITY = 0.98
MIN_TOP_10_OVERLAP = 3
PATH_MAX_RELATIVE_L2 = 0.10
PATH_MIN_COSINE_SIMILARITY = 0.995
PATH_MIN_TOP_10_OVERLAP = 5
CONTEXT_TOKENS = 4_096
CACHE_BLOCKS = 256
KV_CACHE_BYTES = 1_342_177_280
EXPECTED_ARTIFACT_SHA256 = (
    "2243acf1df627ff6ec13840c8ffcf35c77e89205eb36cef7561b85c9c98b9147"
)
EXPECTED_PACKAGE_SHA256 = (
    "c2133dfe392f3d5608bde17ed764ae8347c3096c500a58aa235adbeb63d1a0eb"
)
EXPECTED_SOURCE_METADATA_SHA256 = (
    "1710ebf504c3cf84616f265f57575d48b91804635a0c0151875eadc91fbc122b"
)
EXPECTED_SOURCE_PAYLOAD_MANIFEST_SHA256 = (
    "5972a024c91509b432e68ee39a3dd1cf7a0f0ba2ba48fe7ef5c0bfb02957405c"
)
EXPECTED_SOURCE_SHA256SUMS_SHA256 = (
    "d6d083fc5881480de3a60ae413b56c057d4eaed6d96286db554d0e3c50fecec5"
)
EXPECTED_SOURCE_FIXTURE_SET_ID = "qwen3-14b-fp8-sq8-serving-v0.1"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
F32 = struct.Struct("<f")
U32 = struct.Struct("<I")
CHUNK_ELEMENTS = 16_384


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


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle, object_pairs_hook=reject_duplicate_keys)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        fail(f"failed to read JSON {path}: {error}")
    if not isinstance(value, dict):
        fail(f"JSON root must be an object: {path}")
    return value


def integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        fail(f"{label} must be an integer")
    return value


def finite_float(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        fail(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        fail(f"{label} must be finite")
    return result


def finite_f32(value: Any, label: str) -> float:
    result = finite_float(value, label)
    try:
        return F32.unpack(F32.pack(result))[0]
    except (OverflowError, struct.error):
        fail(f"{label} must be representable as F32")


def sha256_value(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        fail(f"{label} must be a lowercase SHA-256")
    return value


def regular_file(path: Path, expected_bytes: int | None, label: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as error:
        fail(f"failed to stat {label} {path}: {error}")
    if not stat.S_ISREG(metadata.st_mode):
        fail(f"{label} must be a regular file: {path}")
    if expected_bytes is not None and metadata.st_size != expected_bytes:
        fail(f"{label} must be {expected_bytes} bytes: path={path} bytes={metadata.st_size}")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        fail(f"failed to hash {path}: {error}")
    return digest.hexdigest()


def is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def resolve_capture(
    manifest_path: Path,
    value: Any,
    label: str,
    *,
    expected_bytes: int,
    source_root: Path,
) -> Path:
    if not isinstance(value, str) or not value:
        fail(f"{label} must be a nonempty path")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    regular_file(candidate, expected_bytes, label)
    try:
        capture = candidate.resolve(strict=True)
        capture_root = manifest_path.parent.resolve(strict=True)
    except OSError as error:
        fail(f"failed to resolve {label}: {error}")
    if not is_within(capture, capture_root):
        fail(f"{label} must remain inside the capture manifest directory")
    if is_within(capture, source_root):
        fail(f"{label} must remain outside the source oracle directory")
    return capture


def reject_same_file(actual: Path, reference: Path, label: str) -> None:
    try:
        if os.path.samefile(actual, reference):
            fail(f"{label} actual capture aliases the source/reference payload")
    except OSError as error:
        fail(f"failed to compare {label} file identity: {error}")


def push_top(heap: list[tuple[float, int]], value: float, token_id: int) -> None:
    item = (value, -token_id)
    if len(heap) < TOP_K:
        heapq.heappush(heap, item)
    elif item > heap[0]:
        heapq.heapreplace(heap, item)


def top_from_heap(heap: list[tuple[float, int]]) -> list[dict[str, float | int]]:
    return [
        {"token_id": -negative_id, "logit": value}
        for value, negative_id in sorted(heap, key=lambda item: (-item[0], -item[1]))
    ]


def compare_f32_files(
    actual_path: Path,
    reference_path: Path,
    *,
    elements: int,
    top_k: bool,
    label: str,
) -> dict[str, Any]:
    expected_bytes = elements * F32.size
    regular_file(actual_path, expected_bytes, f"{label} actual")
    regular_file(reference_path, expected_bytes, f"{label} reference")
    diff_squared = 0.0
    actual_squared = 0.0
    reference_squared = 0.0
    dot = 0.0
    maximum_absolute_error = 0.0
    actual_top: list[tuple[float, int]] = []
    reference_top: list[tuple[float, int]] = []
    processed = 0
    try:
        with actual_path.open("rb") as actual, reference_path.open("rb") as reference:
            while processed < elements:
                count = min(CHUNK_ELEMENTS, elements - processed)
                actual_bytes = actual.read(count * F32.size)
                reference_bytes = reference.read(count * F32.size)
                if len(actual_bytes) != count * F32.size or len(reference_bytes) != count * F32.size:
                    fail(f"{label} changed during comparison")
                actual_values = struct.iter_unpack("<f", actual_bytes)
                reference_values = struct.iter_unpack("<f", reference_bytes)
                for offset, (actual_item, reference_item) in enumerate(
                    zip(actual_values, reference_values, strict=True)
                ):
                    actual_value = float(actual_item[0])
                    reference_value = float(reference_item[0])
                    token_id = processed + offset
                    if not math.isfinite(actual_value) or not math.isfinite(reference_value):
                        fail(f"{label} contains a non-finite value at {token_id}")
                    difference = actual_value - reference_value
                    diff_squared += difference * difference
                    actual_squared += actual_value * actual_value
                    reference_squared += reference_value * reference_value
                    dot += actual_value * reference_value
                    maximum_absolute_error = max(maximum_absolute_error, abs(difference))
                    if top_k:
                        push_top(actual_top, actual_value, token_id)
                        push_top(reference_top, reference_value, token_id)
                processed += count
            if actual.read(1) or reference.read(1):
                fail(f"{label} grew during comparison")
    except OSError as error:
        fail(f"failed to compare {label}: {error}")
    reference_norm = math.sqrt(reference_squared)
    cosine_denominator = math.sqrt(actual_squared) * reference_norm
    if cosine_denominator == 0.0 or not math.isfinite(cosine_denominator):
        fail(f"{label} has an invalid cosine denominator")
    metrics: dict[str, Any] = {
        "elements": elements,
        "nonfinite_count": 0,
        "relative_l2": math.sqrt(diff_squared) / max(reference_norm, 1e-30),
        "cosine_similarity": dot / cosine_denominator,
        "max_abs": maximum_absolute_error,
    }
    if top_k:
        actual_ranked = top_from_heap(actual_top)
        reference_ranked = top_from_heap(reference_top)
        actual_ids = {item["token_id"] for item in actual_ranked}
        reference_ids = {item["token_id"] for item in reference_ranked}
        metrics.update(
            {
                "actual_top_10": actual_ranked,
                "reference_top_10": reference_ranked,
                "top_1_exact": actual_ranked[0]["token_id"]
                == reference_ranked[0]["token_id"],
                "top_10_overlap": len(actual_ids & reference_ids),
            }
        )
    return metrics


def validate_gate(metrics: dict[str, Any], *, logits: bool, label: str) -> None:
    if (
        metrics["nonfinite_count"] != 0
        or metrics["relative_l2"] > MAX_RELATIVE_L2
        or metrics["cosine_similarity"] < MIN_COSINE_SIMILARITY
    ):
        fail(f"{label} fails the SQ8 source-model numerical gate: {metrics}")
    if logits and (
        metrics["top_1_exact"] is not True
        or metrics["top_10_overlap"] < MIN_TOP_10_OVERLAP
    ):
        fail(f"{label} fails the SQ8 source-model token gate: {metrics}")


def read_first_u32(path: Path) -> int:
    regular_file(path, U32.size, "greedy-g1 oracle")
    try:
        return U32.unpack(path.read_bytes())[0]
    except OSError as error:
        fail(f"failed to read {path}: {error}")


def validate_source_oracle(fixture_root: Path) -> dict[str, Any]:
    try:
        metadata = fixture_root.lstat()
    except OSError as error:
        fail(f"failed to stat source oracle root {fixture_root}: {error}")
    if not stat.S_ISDIR(metadata.st_mode):
        fail(f"source oracle root must be a directory, not a symlink: {fixture_root}")
    try:
        root = fixture_root.resolve(strict=True)
    except OSError as error:
        fail(f"failed to resolve source oracle root {fixture_root}: {error}")

    fixed_files = {
        "metadata.json": EXPECTED_SOURCE_METADATA_SHA256,
        "payload-manifest.json": EXPECTED_SOURCE_PAYLOAD_MANIFEST_SHA256,
        "SHA256SUMS": EXPECTED_SOURCE_SHA256SUMS_SHA256,
    }
    for relative_path, expected_sha256 in fixed_files.items():
        path = regular_file(root / relative_path, None, f"source oracle {relative_path}")
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            fail(
                f"source oracle {relative_path} identity differs: "
                f"expected={expected_sha256} actual={actual_sha256}"
            )

    metadata_json = load_json(root / "metadata.json")
    source_model = metadata_json.get("source_model")
    source_identity = source_model.get("identity") if isinstance(source_model, dict) else None
    source_fixture = metadata_json.get("source_fixture")
    payload_metadata = metadata_json.get("payload_manifest")
    if (
        metadata_json.get("schema_version") != "ullm.sq8.serving_oracle.v1"
        or metadata_json.get("status") != "captured_real_vllm"
        or not isinstance(source_identity, dict)
        or source_identity.get("artifact_content_sha256") != EXPECTED_ARTIFACT_SHA256
        or source_identity.get("package_manifest_sha256") != EXPECTED_PACKAGE_SHA256
        or not isinstance(source_fixture, dict)
        or source_fixture.get("fixture_set_id") != EXPECTED_SOURCE_FIXTURE_SET_ID
        or not isinstance(payload_metadata, dict)
        or payload_metadata.get("sha256") != EXPECTED_SOURCE_PAYLOAD_MANIFEST_SHA256
    ):
        fail("source oracle metadata identity differs")

    manifest = load_json(root / "payload-manifest.json")
    entries = manifest.get("files")
    if (
        manifest.get("schema_version")
        != "ullm.sq8.serving_oracle_payload_manifest.v1"
        or not isinstance(entries, list)
    ):
        fail("source oracle payload manifest contract differs")
    payloads: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(entries):
        label = f"source oracle payload manifest files[{index}]"
        if not isinstance(entry, dict):
            fail(f"{label} must be an object")
        relative_value = entry.get("file")
        if not isinstance(relative_value, str) or not relative_value:
            fail(f"{label}.file must be a nonempty path")
        relative_path = Path(relative_value)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.as_posix() != relative_value
            or relative_value in payloads
        ):
            fail(f"{label}.file is unsafe or duplicated")
        byte_count = integer(entry.get("bytes"), f"{label}.bytes")
        if byte_count < 0:
            fail(f"{label}.bytes must be nonnegative")
        payloads[relative_value] = {
            "bytes": byte_count,
            "sha256": sha256_value(entry.get("sha256"), f"{label}.sha256"),
        }
    return {
        "root": root,
        "payloads": payloads,
        "evidence": {
            "fixture_root": str(root),
            "fixture_set_id": EXPECTED_SOURCE_FIXTURE_SET_ID,
            "metadata_sha256": EXPECTED_SOURCE_METADATA_SHA256,
            "payload_manifest_sha256": EXPECTED_SOURCE_PAYLOAD_MANIFEST_SHA256,
            "sha256sums_sha256": EXPECTED_SOURCE_SHA256SUMS_SHA256,
        },
    }


def source_payload(
    source_oracle: dict[str, Any], relative_path: str, expected_bytes: int
) -> tuple[Path, str]:
    entry = source_oracle["payloads"].get(relative_path)
    if not isinstance(entry, dict) or entry.get("bytes") != expected_bytes:
        fail(
            f"source oracle payload manifest does not bind "
            f"{relative_path} to {expected_bytes} bytes"
        )
    expected_sha256 = entry["sha256"]
    path = regular_file(
        source_oracle["root"] / relative_path,
        expected_bytes,
        f"source oracle payload {relative_path}",
    )
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        fail(
            f"source oracle payload hash differs for {relative_path}: "
            f"expected={expected_sha256} actual={actual_sha256}"
        )
    return path, expected_sha256


def validate_recorded_top1(
    token_value: Any, logit_value: Any, metrics: dict[str, Any], label: str
) -> None:
    ranked = metrics.get("actual_top_10")
    if not isinstance(ranked, list) or not ranked:
        fail(f"{label} has no recomputed top-k values")
    actual = ranked[0]
    recorded_token = integer(token_value, f"{label}.token_id")
    recorded_logit = finite_f32(logit_value, f"{label}.logit")
    if recorded_token != actual["token_id"] or recorded_logit != actual["logit"]:
        fail(
            f"{label} differs from the raw logits: "
            f"recorded=({recorded_token}, {recorded_logit}) "
            f"recomputed=({actual['token_id']}, {actual['logit']})"
        )


def validate_device(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        fail(f"{label} must be an object")
    arch = value.get("gcn_arch_name")
    memory = integer(value.get("total_global_mem"), f"{label}.total_global_mem")
    if (
        value.get("device_id") != 0
        or value.get("backend") != "hip"
        or value.get("name") != "AMD Radeon Graphics"
        or not isinstance(arch, str)
        or (arch and arch.split(":", 1)[0].lower() != "gfx1201")
        or value.get("compute_major") != 12
        or value.get("compute_minor") != 0
        or not (30 * 1024**3 <= memory <= 34 * 1024**3)
    ):
        fail(f"{label} is not the isolated R9700 identity")


def validate_result(
    result_path: Path, source_oracle: dict[str, Any]
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    regular_file(result_path, None, "serving result")
    result = load_json(result_path)
    if result.get("schema_version") != SCHEMA_VERSION:
        fail(f"{result_path} has the wrong schema version")
    if result.get("artifact_content_sha256") != EXPECTED_ARTIFACT_SHA256:
        fail(f"{result_path} has the wrong artifact identity")
    if result.get("package_manifest_sha256") != EXPECTED_PACKAGE_SHA256:
        fail(f"{result_path} has the wrong package identity")
    validate_device(result.get("device"), f"{result_path}.device")
    if (
        result.get("kv_cache_bytes") != KV_CACHE_BYTES
        or result.get("cache_blocks") != CACHE_BLOCKS
        or result.get("context_tokens") != CONTEXT_TOKENS
        or result.get("post_reset_status") != "ready"
        or result.get("post_reset_active") != 0
        or result.get("post_reset_waiting") != 0
        or result.get("post_reset_allocated_blocks") != 0
        or result.get("post_reset_cache_lengths_all_zero") is not True
    ):
        fail(f"{result_path} does not prove the post-reset serving baseline")
    requests = result.get("requests")
    if not isinstance(requests, list) or not requests:
        fail(f"{result_path}.requests must be nonempty")
    prompt_results: dict[int, dict[str, Any]] = {}
    for request_index, request in enumerate(requests):
        label = f"{result_path}.requests[{request_index}]"
        if not isinstance(request, dict):
            fail(f"{label} must be an object")
        prompt = request.get("prompt_token_ids")
        if not isinstance(prompt, list) or any(
            integer(token, f"{label}.prompt_token_ids") != expected
            for expected, token in enumerate(prompt, start=1)
        ):
            fail(f"{label} does not use the ascending raw-token fixture")
        prompt_length = len(prompt)
        if prompt_length not in PROMPT_LENGTHS or prompt_length in prompt_results:
            fail(f"{label} has an unexpected or duplicate prompt length {prompt_length}")
        generated = request.get("generated_token_ids")
        capture = request.get("oracle_capture")
        if request.get("max_new_tokens") != 1 or not isinstance(generated, list) or len(generated) != 1:
            fail(f"{label} must be a G=1 oracle request")
        if not isinstance(capture, dict) or capture.get("position") != prompt_length - 1:
            fail(f"{label} has invalid oracle capture metadata")
        prompt_id = f"raw-p{prompt_length:04d}"
        reference_prefix = f"prompts/{prompt_id}"
        reference_hidden, reference_hidden_hash = source_payload(
            source_oracle,
            f"{reference_prefix}/final-hidden.f32le",
            HIDDEN_SIZE * F32.size,
        )
        reference_logits, reference_logits_hash = source_payload(
            source_oracle,
            f"{reference_prefix}/prefill-logits.f32le",
            VOCAB_SIZE * F32.size,
        )
        reference_greedy, reference_greedy_hash = source_payload(
            source_oracle,
            f"{reference_prefix}/greedy-g1.u32le",
            U32.size,
        )
        expected_token = read_first_u32(reference_greedy)
        actual_token = integer(generated[0], f"{label}.generated_token_ids[0]")
        if actual_token != expected_token or capture.get("top1_token_id") != expected_token:
            fail(f"{label} first token does not match the source oracle")
        final_hidden_path = resolve_capture(
            result_path,
            capture.get("final_hidden_file"),
            f"{label}.final_hidden_file",
            expected_bytes=HIDDEN_SIZE * F32.size,
            source_root=source_oracle["root"],
        )
        logits_path = resolve_capture(
            result_path,
            capture.get("logits_file"),
            f"{label}.logits_file",
            expected_bytes=VOCAB_SIZE * F32.size,
            source_root=source_oracle["root"],
        )
        expected_hidden_hash = sha256_value(
            capture.get("final_hidden_f32_le_sha256"),
            f"{label}.final_hidden_f32_le_sha256",
        )
        expected_logits_hash = sha256_value(
            capture.get("logits_f32_le_sha256"), f"{label}.logits_f32_le_sha256"
        )
        if sha256_file(final_hidden_path) != expected_hidden_hash:
            fail(f"{label} final-hidden hash differs")
        if sha256_file(logits_path) != expected_logits_hash:
            fail(f"{label} logits hash differs")
        reject_same_file(final_hidden_path, reference_hidden, f"{prompt_id} final hidden")
        reject_same_file(logits_path, reference_logits, f"{prompt_id} logits")
        hidden_metrics = compare_f32_files(
            final_hidden_path,
            reference_hidden,
            elements=HIDDEN_SIZE,
            top_k=False,
            label=f"{prompt_id} final hidden",
        )
        logits_metrics = compare_f32_files(
            logits_path,
            reference_logits,
            elements=VOCAB_SIZE,
            top_k=True,
            label=f"{prompt_id} logits",
        )
        validate_gate(hidden_metrics, logits=False, label=f"{prompt_id} final hidden")
        validate_gate(logits_metrics, logits=True, label=f"{prompt_id} logits")
        validate_recorded_top1(
            capture.get("top1_token_id"),
            capture.get("top1_logit"),
            logits_metrics,
            f"{label}.oracle_capture.top1",
        )
        prompt_results[prompt_length] = {
            "prompt_id": prompt_id,
            "prompt_tokens": prompt_length,
            "generated_token_id": actual_token,
            "final_hidden": hidden_metrics,
            "logits": logits_metrics,
            "capture": {
                "final_hidden_file": str(final_hidden_path),
                "final_hidden_sha256": expected_hidden_hash,
                "logits_file": str(logits_path),
                "logits_sha256": expected_logits_hash,
            },
            "source_reference": {
                "final_hidden_sha256": reference_hidden_hash,
                "logits_sha256": reference_logits_hash,
                "greedy_g1_sha256": reference_greedy_hash,
            },
        }
    return prompt_results, {
        "result_file": str(result_path),
        "result_sha256": sha256_file(result_path),
    }


def validate_p7_path_equivalence(
    manifest_path: Path, p8_prompt: dict[str, Any], source_root: Path
) -> dict[str, Any]:
    regular_file(manifest_path, None, "P7 first-step capture manifest")
    manifest = load_json(manifest_path)
    if (
        manifest.get("schema_version") != "ullm.sq8.p7_first_step_capture.v1"
        or manifest.get("prompt_token_ids") != list(range(1, 9))
        or manifest.get("step_index") != 0
        or manifest.get("output_token_id") != p8_prompt["generated_token_id"]
    ):
        fail("P7 first-step capture manifest contract differs")
    p7_hidden = resolve_capture(
        manifest_path,
        manifest.get("final_hidden_file"),
        "P7 first-step final_hidden_file",
        expected_bytes=HIDDEN_SIZE * F32.size,
        source_root=source_root,
    )
    p7_logits = resolve_capture(
        manifest_path,
        manifest.get("logits_file"),
        "P7 first-step logits_file",
        expected_bytes=VOCAB_SIZE * F32.size,
        source_root=source_root,
    )
    p7_hidden_hash = sha256_value(
        manifest.get("final_hidden_f32_le_sha256"),
        "P7 first-step final_hidden_f32_le_sha256",
    )
    p7_logits_hash = sha256_value(
        manifest.get("logits_f32_le_sha256"), "P7 first-step logits_f32_le_sha256"
    )
    if sha256_file(p7_hidden) != p7_hidden_hash or sha256_file(p7_logits) != p7_logits_hash:
        fail("P7 first-step capture hash differs")
    p8_capture = p8_prompt["capture"]
    reject_same_file(
        p7_hidden,
        Path(p8_capture["final_hidden_file"]),
        "P7 M=8 against P8 M=1 final hidden",
    )
    reject_same_file(
        p7_logits,
        Path(p8_capture["logits_file"]),
        "P7 M=8 against P8 M=1 logits",
    )
    hidden_metrics = compare_f32_files(
        p7_hidden,
        Path(p8_capture["final_hidden_file"]),
        elements=HIDDEN_SIZE,
        top_k=False,
        label="P7 M=8 against P8 M=1 final hidden",
    )
    logits_metrics = compare_f32_files(
        p7_logits,
        Path(p8_capture["logits_file"]),
        elements=VOCAB_SIZE,
        top_k=True,
        label="P7 M=8 against P8 M=1 logits",
    )
    for label, metrics in (("final hidden", hidden_metrics), ("logits", logits_metrics)):
        if (
            metrics["relative_l2"] > PATH_MAX_RELATIVE_L2
            or metrics["cosine_similarity"] < PATH_MIN_COSINE_SIMILARITY
        ):
            fail(f"P7/P8 {label} path-equivalence gate failed: {metrics}")
    if (
        logits_metrics["top_1_exact"] is not True
        or logits_metrics["top_10_overlap"] < PATH_MIN_TOP_10_OVERLAP
    ):
        fail(f"P7/P8 logits token path-equivalence gate failed: {logits_metrics}")
    validate_recorded_top1(
        manifest.get("output_token_id"),
        manifest.get("output_logit"),
        logits_metrics,
        "P7 first-step output",
    )
    return {
        "prompt_tokens": 8,
        "p7_mode": "M=8_paged_prefill",
        "p8_mode": "sequential_M=1_prefill",
        "thresholds": {
            "max_relative_l2": PATH_MAX_RELATIVE_L2,
            "min_cosine_similarity": PATH_MIN_COSINE_SIMILARITY,
            "top_1": "exact",
            "min_top_10_overlap": PATH_MIN_TOP_10_OVERLAP,
        },
        "final_hidden": hidden_metrics,
        "logits": logits_metrics,
        "p7_capture_manifest": str(manifest_path),
        "p7_capture_manifest_sha256": sha256_file(manifest_path),
    }


def validate_results(
    result_paths: list[Path],
    fixture_root: Path,
    p7_capture_manifest: Path | None = None,
) -> dict[str, Any]:
    source_oracle = validate_source_oracle(fixture_root)
    combined: dict[int, dict[str, Any]] = {}
    evidence = []
    for result_path in result_paths:
        values, result_evidence = validate_result(result_path, source_oracle)
        overlap = set(combined) & set(values)
        if overlap:
            fail(f"duplicate prompt lengths across results: {sorted(overlap)}")
        combined.update(values)
        evidence.append(result_evidence)
    if set(combined) != set(PROMPT_LENGTHS):
        fail(
            f"runtime oracle prompt coverage differs: "
            f"expected={list(PROMPT_LENGTHS)} actual={sorted(combined)}"
        )
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "passed": True,
        "thresholds": {
            "max_relative_l2": MAX_RELATIVE_L2,
            "min_cosine_similarity": MIN_COSINE_SIMILARITY,
            "top_1": "exact",
            "min_top_10_overlap": MIN_TOP_10_OVERLAP,
        },
        "source_oracle": source_oracle["evidence"],
        "prompts": [combined[length] for length in PROMPT_LENGTHS],
        "evidence": evidence,
    }
    if p7_capture_manifest is not None:
        result["p7_path_equivalence"] = validate_p7_path_equivalence(
            p7_capture_manifest, combined[8], source_oracle["root"]
        )
    return result


def write_json_create_new(path: Path, value: dict[str, Any]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as error:
        fail(f"failed to create validation output {path}: {error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument(
        "--fixture-root",
        type=Path,
        default=Path("tests/fixtures/sq8-serving-v0.1/oracles/vllm-source-v0.1"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--p7-capture", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        validation = validate_results(args.results, args.fixture_root, args.p7_capture)
        if args.output is not None:
            write_json_create_new(args.output, validation)
    except ValidationError as error:
        print(f"validation failed: {error}", file=sys.stderr)
        return 1
    worst_relative_l2 = max(
        max(prompt["final_hidden"]["relative_l2"], prompt["logits"]["relative_l2"])
        for prompt in validation["prompts"]
    )
    minimum_cosine = min(
        min(
            prompt["final_hidden"]["cosine_similarity"],
            prompt["logits"]["cosine_similarity"],
        )
        for prompt in validation["prompts"]
    )
    print(
        f"passed=true prompts={list(PROMPT_LENGTHS)} "
        f"worst_relative_l2={worst_relative_l2:.9f} "
        f"minimum_cosine={minimum_cosine:.9f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
