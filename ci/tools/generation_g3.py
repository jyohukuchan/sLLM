#!/usr/bin/env python3
"""Fail-closed G3 run, evidence-manifest, and aggregate controller.

The only record accepted by ``normalize`` and ``aggregate`` is a manifest
created by ``run``.  A manifest is the bounded bridge between the files which
must remain outside tracked evidence (the executable and raw report) and the
path-free normalized row.  The test-only fixture entry points are deliberately
named and are not exposed by the production CLI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

try:
    from jsonschema import Draft202012Validator, FormatChecker
except ImportError as exc:  # pragma: no cover - pinned host dependency is required
    Draft202012Validator = None  # type: ignore[assignment,misc]
    FormatChecker = None  # type: ignore[assignment,misc]
    _JSONSCHEMA_IMPORT_ERROR = exc
else:
    _JSONSCHEMA_IMPORT_ERROR = None


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATRIX = ROOT / "ci/matrix/generation-g3-v1.json"
REPORT_SCHEMA = ROOT / "ci/schema/generation-g3-report-v1.schema.json"
AGGREGATE_SCHEMA = ROOT / "ci/schema/generation-g3-aggregate-v1.schema.json"
RAW_REPORT_SCHEMA = ROOT / "ci/schema/model-frontend-cli-report-v1.schema.json"
MATRIX_RELATIVE = "ci/matrix/generation-g3-v1.json"
MODEL_LOCK_PATH = ROOT / "docs/models/locks/qwen3.5-4b-bf16.json"
MODEL_CACHE_PATH = Path(
    "/home/homelab1/.cache/sllm/models/Qwen--Qwen3.5-4B/"
    "snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
)
UNICODE_MESSAGE = "user:こんにちは。答えは「はい」の一語だけにしてください。"

SCHEMA_VERSION = "generation-g3-matrix-v1"
REPORT_VERSION = "generation-g3-report-v1"
AGGREGATE_VERSION = "generation-g3-aggregate-v1"
MANIFEST_KIND = "evidence_manifest"
ROW_KIND = "normalized_row"
MODEL = {
    "repo_id": "Qwen/Qwen3.5-4B",
    "resolved_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "lock_fingerprint": "sha256:f143d7b504170d071c77818105f7a07dc0297c6bea0c61a5404b071fed0c1fae",
}
CANDIDATE = {
    "commit": "2c859a579007c388d956e46905ef71fb5a9d5881",
    "tree": "1fb63550fe2d4405a97a00ba28e2bcb1dfc75f14",
}
PLAN_DIGEST = "sha256:0474ed893fbc043c3ace0197515f8d99e27fe3d28a4844fbdd9781bb9d30c7fa"
CANONICAL_MATRIX_SHA256 = "b0e3bc7d31cf8084bb8e3e5c66767353eb5e75992b72a4edd977e97656184f39"
TARGETS = ("gfx1030", "gfx1201")
TARGET_IDENTITIES = {
    "gfx1030": {"gpu_uuid": "GPU-76a08c022586fed6", "gpu_bdf": "0000:03:00.0"},
    "gfx1201": {"gpu_uuid": "GPU-a8e9ddefa2d60f55", "gpu_bdf": "0000:07:00.0"},
}
STOP_TOKEN = 248046
FORBIDDEN_GPU_GOLDEN_STOP_TOKEN = 248044
TIMEOUT_SECONDS = 180
TIMEOUT_NS = TIMEOUT_SECONDS * 1_000_000_000
TERMINATION_GRACE_SECONDS = 2
POST_OBSERVATION_ATTEMPTS = 21
POST_OBSERVATION_INTERVAL_SECONDS = 0.25
MAX_MATRIX_BYTES = 1 * 1024 * 1024
MAX_SCHEMA_BYTES = 2 * 1024 * 1024
MAX_REPORT_BYTES = 32 * 1024 * 1024
MAX_BINARY_BYTES = 4 * 1024 * 1024 * 1024
MAX_INPUT_TOKENS = 257
MAX_GENERATED_TOKENS = 4096
MAX_COMMAND_ITEMS = 64
MAX_COMMAND_ITEM_BYTES = 4096
MAX_PROCESS_RECORDS = 1024
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
AMD_SMI_EXECUTABLE = "/opt/rocm/core-7.14/bin/amd-smi"
VISIBILITY_NAMES = (
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "ROCR_VISIBLE_DEVICES",
)

SHA40_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FINGERPRINT_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
RUN_ID_RE = re.compile(r"^g3-run-[0-9a-f]{32}$")

CASE_IDS = (
    "g3-prompt-1",
    "g3-prompt-7",
    "g3-prompt-255",
    "g3-prompt-256",
    "g3-prompt-257",
    "g3-unicode-chat-248046",
)
CASE_INPUT_LENGTHS = (1, 7, 255, 256, 257, 24)
CASE_DESCRIPTORS = (
    "ascii-hello",
    "ascii-a-space-joined-7",
    "ascii-a-space-joined-255",
    "ascii-a-space-joined-256",
    "ascii-a-space-joined-257",
    "reviewed-unicode-chat",
)
PROMPT_LENGTHS = (1, 7, 255, 256, 257)
MATRIX_KEYS = {
    "schema_version", "matrix_id", "suite_id", "tier", "required", "timeout_seconds",
    "candidate", "plan_digest", "model", "stop_policy", "targets", "cases",
}
TARGET_KEYS = {
    "order", "row_id", "target", "backend", "gpu_uuid", "gpu_bdf",
    "binary_sha256", "cases",
}
CASE_KEYS = {"order", "id", "input_kind", "input_token_length", "descriptor", "golden"}
GOLDEN_KEYS = {
    "status", "input_token_spec", "generated_token_ids", "visible_token_ids",
    "decode_input_token_ids", "output_text", "stop_reason", "audit", "pending_reason",
}
STOP_REASON_KEYS = {"version", "reason_version", "kind", "token_id"}
GOLDEN_AUDIT_KEYS = {"prefill_tokens", "decode_steps", "submission_count", "kernel_dispatch_count"}


class G3Error(ValueError):
    """A malformed, stale, mixed, or otherwise non-PASS G3 contract."""


def _fail(message: str) -> None:
    raise G3Error(message)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_keys(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, dict):
        _fail(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        detail = []
        if missing:
            detail.append("missing " + ",".join(missing))
        if extra:
            detail.append("unknown " + ",".join(extra))
        _fail(f"{label} is not closed: {'; '.join(detail)}")


def _regular_file(path: Path, label: str, max_bytes: int | None = None) -> Path:
    try:
        if path.is_symlink() or not path.is_file():
            _fail(f"{label} must be a regular non-symlink file: {path}")
        size = path.stat().st_size
    except OSError as exc:
        _fail(f"cannot inspect {label} {path}: {exc}")
    if max_bytes is not None and size > max_bytes:
        _fail(f"{label} exceeds bounded size {max_bytes}: {path}")
    return path


def _reject_json_constant(value: str) -> Any:
    _fail(f"non-standard JSON number is forbidden: {value}")


def _parse_json_bytes(data: bytes, label: str) -> Any:
    try:
        text = data.decode("utf-8")

        def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    _fail(f"duplicate JSON key in {label}: {key}")
                result[key] = value
            return result

        return json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except G3Error:
        raise
    except (UnicodeError, ValueError) as exc:
        _fail(f"cannot parse JSON {label}: {exc}")


def _read_json(path: Path, label: str, max_bytes: int) -> Any:
    _regular_file(path, label, max_bytes)
    try:
        data = path.read_bytes()
    except OSError as exc:
        _fail(f"cannot read JSON {label} {path}: {exc}")
    return _parse_json_bytes(data, label)


def _read_hashed_json(path: Path, label: str, max_bytes: int) -> tuple[Any, str, bytes]:
    _regular_file(path, label, max_bytes)
    try:
        data = path.read_bytes()
    except OSError as exc:
        _fail(f"cannot read {label} {path}: {exc}")
    return _parse_json_bytes(data, label), _sha256_bytes(data), data


def _canonical_bytes(value: Any) -> bytes:
    try:
        return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        _fail(f"value is not canonical JSON: {exc}")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path, label: str, max_bytes: int | None = None) -> str:
    _regular_file(path, label, max_bytes)
    try:
        with path.open("rb") as stream:
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        _fail(f"cannot hash {label} {path}: {exc}")
    result = digest.hexdigest()
    if result == "0" * 64:
        _fail(f"{label} hash is zero")
    return result


def _sha40(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA40_RE.fullmatch(value) is None:
        _fail(f"{label} must be a lowercase 40-character SHA")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None or value == "0" * 64:
        _fail(f"{label} must be a nonzero lowercase 64-character SHA")
    return value


def _fingerprint(value: Any, label: str) -> str:
    if not isinstance(value, str) or FINGERPRINT_RE.fullmatch(value) is None:
        _fail(f"{label} must be a sha256 fingerprint")
    return value


def _token_ids(value: Any, label: str, *, max_items: int = MAX_GENERATED_TOKENS) -> list[int]:
    if not isinstance(value, list) or len(value) > max_items:
        _fail(f"{label} must be a bounded token ID array")
    result: list[int] = []
    for index, token in enumerate(value):
        if not _is_int(token) or not 0 <= token <= 4_294_967_295:
            _fail(f"{label}[{index}] is not a valid token ID")
        result.append(token)
    return result


def _schema_validator(path: Path, label: str) -> Any:
    if Draft202012Validator is None:
        _fail(f"jsonschema is required for {label}: {_JSONSCHEMA_IMPORT_ERROR}")
    schema = _read_json(path, label, MAX_SCHEMA_BYTES)
    try:
        Draft202012Validator.check_schema(schema)
        return Draft202012Validator(schema, format_checker=FormatChecker())
    except Exception as exc:  # jsonschema has several schema-error classes
        _fail(f"{label} is not a valid Draft 2020-12 schema: {exc}")


def _schema_validate(value: Any, path: Path, label: str) -> None:
    validator = _schema_validator(path, label + " schema")
    errors = sorted(validator.iter_errors(value), key=lambda error: list(error.absolute_path))
    if errors:
        first = errors[0]
        _fail(f"{label} fails checked-in schema: {list(first.absolute_path)}: {first.message}")


def _matrix_case_map(matrix: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {case["id"]: case for case in matrix["cases"]}


def _matrix_target_map(matrix: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {target["target"]: target for target in matrix["targets"]}


def _contains_value(value: Any, needle: int) -> bool:
    if isinstance(value, dict):
        return any(_contains_value(item, needle) for item in value.values())
    if isinstance(value, list):
        return any(_contains_value(item, needle) for item in value)
    return value == needle


def _expand_input_spec(value: Any, label: str) -> list[int]:
    if not isinstance(value, dict) or "kind" not in value:
        _fail(f"{label} must be a closed token specification")
    if value["kind"] == "literal":
        _require_keys(value, {"kind", "token_ids"}, label)
        return _token_ids(value["token_ids"], label + " token_ids", max_items=MAX_INPUT_TOKENS)
    if value["kind"] == "head-repeat":
        _require_keys(value, {"kind", "head_token_id", "repeat_token_id", "repeat_count"}, label)
        head = _token_ids([value["head_token_id"]], label + " head", max_items=1)[0]
        repeat = _token_ids([value["repeat_token_id"]], label + " repeat", max_items=1)[0]
        count = value["repeat_count"]
        if not _is_int(count) or not 0 <= count < MAX_INPUT_TOKENS:
            _fail(f"{label} repeat_count is invalid")
        return [head, *([repeat] * count)]
    _fail(f"{label} kind is invalid")


def _validate_stop_reason(value: Any, label: str, *, allow_unreviewed_stop: bool = True) -> dict[str, Any]:
    _require_keys(value, STOP_REASON_KEYS, label)
    if value["version"] != 1 or value["reason_version"] != 1:
        _fail(f"{label} version is not 1")
    if value["kind"] not in {"stop_token", "max_new_tokens"}:
        _fail(f"{label} kind is invalid")
    token_id = value["token_id"]
    if token_id is not None and (not _is_int(token_id) or not 0 <= token_id <= 4_294_967_295):
        _fail(f"{label} token_id is invalid")
    if value["kind"] == "max_new_tokens" and token_id is not None:
        _fail(f"{label} max_new_tokens must have a null token_id")
    if value["kind"] == "stop_token":
        allowed = {STOP_TOKEN, FORBIDDEN_GPU_GOLDEN_STOP_TOKEN} if allow_unreviewed_stop else {STOP_TOKEN}
        if token_id not in allowed:
            _fail(f"{label} uses an unapproved stop token")
    return dict(value)


def _validate_matrix_structure(matrix: dict[str, Any], *, canonical: bool) -> dict[str, Any]:
    _require_keys(matrix, MATRIX_KEYS, "G3 matrix")
    if matrix["schema_version"] != SCHEMA_VERSION or matrix["matrix_id"] != "generation-g3-v1":
        _fail("G3 matrix schema or matrix identity is stale")
    if matrix["suite_id"] != "g3-qwen35-4b-text-generation" or matrix["tier"] != "tier_g3" or matrix["required"] is not True:
        _fail("G3 matrix suite, tier, or required flag is invalid")
    if matrix["timeout_seconds"] != TIMEOUT_SECONDS:
        _fail("G3 matrix timeout must be exactly 180 seconds")
    _require_keys(matrix["candidate"], {"commit", "tree"}, "G3 candidate")
    _sha40(matrix["candidate"]["commit"], "G3 candidate commit")
    _sha40(matrix["candidate"]["tree"], "G3 candidate tree")
    _fingerprint(matrix["plan_digest"], "G3 plan digest")
    _require_keys(matrix["model"], set(MODEL), "G3 model")
    if matrix["model"]["repo_id"] != MODEL["repo_id"]:
        _fail("G3 matrix model repository is not the locked Qwen3.5-4B")
    _sha40(matrix["model"]["resolved_revision"], "G3 model revision")
    _fingerprint(matrix["model"]["lock_fingerprint"], "G3 model lock fingerprint")
    _require_keys(matrix["stop_policy"], {"version", "reviewed_gpu_stop_token_ids", "visible_stop_tokens"}, "G3 stop policy")
    if matrix["stop_policy"] != {"version": 1, "reviewed_gpu_stop_token_ids": [STOP_TOKEN], "visible_stop_tokens": False}:
        _fail("G3 stop policy is not the reviewed GPU policy")
    if _contains_value(matrix, FORBIDDEN_GPU_GOLDEN_STOP_TOKEN):
        _fail("G3 matrix contains a forbidden 248044 GPU golden")

    if not isinstance(matrix["targets"], list) or len(matrix["targets"]) != 2:
        _fail("G3 matrix must select exactly two targets")
    target_map: dict[str, dict[str, Any]] = {}
    for order, target in enumerate(matrix["targets"]):
        _require_keys(target, TARGET_KEYS, f"G3 target {order}")
        name = TARGETS[order]
        if target["order"] != order or target["target"] != name or target["row_id"] != f"generation-g3-{name}":
            _fail("G3 targets are not exactly ordered gfx1030/gfx1201 rows")
        if target["backend"] != "hip" or {key: target[key] for key in ("gpu_uuid", "gpu_bdf")} != TARGET_IDENTITIES[name]:
            _fail(f"G3 {name} GPU identity or backend is stale")
        _sha256(target["binary_sha256"], f"G3 {name} binary_sha256")
        if target["target"] in target_map or target["cases"] != list(CASE_IDS):
            _fail("G3 target cases are missing, duplicated, or stale")
        target_map[name] = target

    if not isinstance(matrix["cases"], list) or len(matrix["cases"]) != len(CASE_IDS):
        _fail("G3 matrix must contain exactly six cases")
    case_map: dict[str, dict[str, Any]] = {}
    for order, case in enumerate(matrix["cases"]):
        _require_keys(case, CASE_KEYS, f"G3 case {order}")
        case_id = case["id"]
        if case["order"] != order or case_id != CASE_IDS[order] or case_id in case_map:
            _fail("G3 cases are not deterministic or contain duplicates")
        if case["input_kind"] not in {"prompt", "messages"} or case["descriptor"] != CASE_DESCRIPTORS[order]:
            _fail(f"G3 case {case_id} input descriptor is invalid")
        expected_length = CASE_INPUT_LENGTHS[order]
        if case["input_token_length"] != expected_length:
            _fail(f"G3 case {case_id} has the wrong token length")
        if order < len(PROMPT_LENGTHS) and case["input_kind"] != "prompt":
            _fail(f"G3 case {case_id} must be a prompt case")
        if order == len(PROMPT_LENGTHS) and case["input_kind"] != "messages":
            _fail("G3 Unicode case must be text-only messages")

        golden = case["golden"]
        expected_golden_keys = GOLDEN_KEYS - ({"pending_reason"} if isinstance(golden, dict) and golden.get("status") == "reviewed" else set())
        _require_keys(golden, expected_golden_keys, f"G3 {case_id} golden")
        if golden["status"] not in {"reviewed", "pending"}:
            _fail(f"G3 {case_id} golden status is invalid")
        if _contains_value(golden, FORBIDDEN_GPU_GOLDEN_STOP_TOKEN):
            _fail(f"G3 {case_id} contains a forbidden 248044 GPU golden")
        input_ids = _expand_input_spec(golden["input_token_spec"], f"G3 {case_id} input_token_spec")
        if len(input_ids) != expected_length:
            _fail(f"G3 case {case_id} input token specification has the wrong length")
        for field in ("generated_token_ids", "visible_token_ids", "decode_input_token_ids"):
            if golden[field] is not None:
                _token_ids(golden[field], f"G3 {case_id} golden {field}")
        if golden["output_text"] is not None and not isinstance(golden["output_text"], str):
            _fail(f"G3 {case_id} golden output_text must be a string or null")
        if golden["status"] == "pending":
            if not isinstance(golden["pending_reason"], str) or not golden["pending_reason"]:
                _fail(f"G3 {case_id} pending golden needs a reason")
            if any(golden[field] is not None for field in ("generated_token_ids", "visible_token_ids", "decode_input_token_ids", "output_text", "stop_reason", "audit")):
                _fail(f"G3 {case_id} pending generation golden must not contain an asserted value")
        else:
            if any(golden[field] is None for field in ("generated_token_ids", "visible_token_ids", "decode_input_token_ids", "output_text", "stop_reason", "audit")):
                _fail(f"G3 {case_id} reviewed golden is incomplete")
            _require_keys(golden["audit"], GOLDEN_AUDIT_KEYS, f"G3 {case_id} golden audit")
            if golden["audit"]["prefill_tokens"] != expected_length:
                _fail(f"G3 {case_id} golden prefill count is invalid")
            for field in GOLDEN_AUDIT_KEYS:
                if not _is_int(golden["audit"][field]) or golden["audit"][field] < 0:
                    _fail(f"G3 {case_id} golden audit {field} is invalid")
            if golden["audit"]["submission_count"] < 1 or golden["audit"]["kernel_dispatch_count"] < 1:
                _fail(f"G3 {case_id} golden dispatch audit is empty")
            stop = _validate_stop_reason(golden["stop_reason"], f"G3 {case_id} golden stop_reason", allow_unreviewed_stop=False)
            generated = golden["generated_token_ids"]
            visible = golden["visible_token_ids"]
            decode_input = golden["decode_input_token_ids"]
            if not generated or len(generated) > MAX_GENERATED_TOKENS:
                _fail(f"G3 {case_id} reviewed golden generated sequence is empty or unbounded")
            if stop["kind"] == "stop_token":
                if generated[-1] != stop["token_id"] or visible != generated[:-1] or decode_input != generated[:-1]:
                    _fail(f"G3 {case_id} stop-token golden has inconsistent token sequences")
            elif visible != generated or decode_input != generated[:-1]:
                _fail(f"G3 {case_id} max-token golden has inconsistent token sequences")
        case_map[case_id] = case
    if set(target_map) != set(TARGETS) or set(case_map) != set(CASE_IDS):
        _fail("G3 matrix selection is empty or incomplete")
    if canonical:
        if matrix["candidate"] != CANDIDATE or matrix["plan_digest"] != PLAN_DIGEST or matrix["model"] != MODEL:
            _fail("G3 canonical candidate, plan, or model identity is stale")
    return matrix


def validate_matrix(path: Path = DEFAULT_MATRIX, *, test_only_fixture_matrix: bool = False) -> dict[str, Any]:
    """Validate the exact canonical matrix, or an explicitly opted-in fixture."""

    path = Path(path)
    matrix = _read_json(path, "G3 matrix", MAX_MATRIX_BYTES)
    actual_hash = _sha256_file(path, "G3 matrix", MAX_MATRIX_BYTES)
    canonical_path = path.resolve() == DEFAULT_MATRIX.resolve()
    if not test_only_fixture_matrix:
        if not canonical_path:
            _fail("production G3 CLI rejects alternate matrix paths")
        if actual_hash != CANONICAL_MATRIX_SHA256:
            _fail("canonical G3 matrix content hash is not pinned")
    return _validate_matrix_structure(matrix, canonical=not test_only_fixture_matrix)


def test_only_validate_fixture_matrix(path: Path) -> dict[str, Any]:
    """Explicit test-only opt-in for a regular fixture matrix."""

    return validate_matrix(path, test_only_fixture_matrix=True)


def _candidate(commit: str, tree: str) -> dict[str, str]:
    commit = _sha40(commit, "candidate commit")
    tree = _sha40(tree, "candidate tree")
    base = {"commit": commit, "tree": tree}
    return {**base, "candidate_sha256": _sha256_bytes(_canonical_bytes(base))}


def _candidate_document(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, dict):
        _fail(f"{label} must be an object")
    if set(value) == {"commit", "tree"}:
        return _candidate(value["commit"], value["tree"])
    if set(value) == {"commit", "tree", "candidate_sha256"}:
        computed = _candidate(value["commit"], value["tree"])
        if computed != value:
            _fail(f"{label} candidate_sha256 is stale")
        return computed
    _require_keys(value, {"commit", "tree"}, label)
    raise AssertionError("unreachable")


def _check_raw_report(raw: dict[str, Any], raw_path: Path, raw_bytes: bytes) -> dict[str, Any]:
    if len(raw_bytes.splitlines()) != 1:
        _fail("model-frontend raw report must contain exactly one JSON line")
    _schema_validate(raw, RAW_REPORT_SCHEMA, "model-frontend raw report")
    if raw.get("schema_version") != "model-frontend-cli-report-v1" or raw.get("command") != "generate" or raw.get("state") != "PASS":
        _fail(f"raw report is not a PASS generate report: {raw_path}")
    return raw["result"]


def _expected_case(matrix: dict[str, Any], case_id: str) -> dict[str, Any]:
    cases = _matrix_case_map(matrix)
    if case_id not in cases:
        _fail(f"unknown G3 case: {case_id}")
    return cases[case_id]


def _validate_raw_semantics(raw: dict[str, Any], result: dict[str, Any], case: dict[str, Any], target: str, matrix: dict[str, Any]) -> None:
    if raw["model"] != matrix["model"]:
        _fail("raw report model revision or lock fingerprint is stale")
    if raw["scope"] != {"offline": True, "gpu_execution": True, "model_execution": True, "generation": True}:
        _fail("raw report scope is not the required offline GPU generation scope")
    execution = result["execution"]
    if execution["target"] != target or execution["selected_backend"] != "hip":
        _fail("raw report target/backend audit is mixed or non-HIP")
    if execution["model_fingerprint"] != matrix["model"]["lock_fingerprint"]:
        _fail("raw report execution model fingerprint is stale")
    if execution["plan_digest"] != matrix["plan_digest"] or execution["device_index"] != 0:
        _fail("raw report plan digest or logical device index is stale")
    if execution["fallback_used"] is not False or execution["all_dispatches_hip"] is not True:
        _fail("raw report contains fallback or non-HIP dispatch evidence")
    if execution["submission_count"] < 1 or execution["kernel_dispatch_count"] < 1:
        _fail("raw report has an empty dispatch selection")
    input_ids = _token_ids(result["input_token_ids"], "raw input_token_ids", max_items=MAX_INPUT_TOKENS)
    generated = _token_ids(result["generated_token_ids"], "raw generated_token_ids")
    visible = _token_ids(result["visible_token_ids"], "raw visible_token_ids")
    decode_input = _token_ids(result["decode_input_token_ids"], "raw decode_input_token_ids")
    if not input_ids or len(input_ids) > MAX_INPUT_TOKENS or not generated or len(generated) > MAX_GENERATED_TOKENS:
        _fail("raw token selection is empty or exceeds the G3 bound")
    if result["input_kind"] != case["input_kind"] or len(input_ids) != case["input_token_length"]:
        _fail("raw report input kind or token length does not match the G3 case")
    if input_ids != _expand_input_spec(case["golden"]["input_token_spec"], f"G3 {case['id']} input_token_spec"):
        _fail("raw report input token IDs do not match the deterministic G3 case")
    if execution["prefill_tokens"] != len(input_ids) or execution["decode_steps"] != len(decode_input):
        _fail("raw report prefill/decode audit does not match token evidence")
    if not isinstance(result["output_text"], str) or len(result["output_text"].encode("utf-8")) > 16 * 1024 * 1024:
        _fail("raw output text is not bounded UTF-8 text")
    stop = _validate_stop_reason(result["stop_reason"], "raw stop_reason")
    if stop["kind"] == "stop_token":
        if generated[-1] != stop["token_id"] or visible != generated[:-1] or decode_input != generated[:-1]:
            _fail("raw stop-token evidence has inconsistent generated/visible/decode sequences")
    elif visible != generated or decode_input != generated[:-1]:
        _fail("raw max-token evidence has inconsistent generated/visible/decode sequences")
    if result["cleanup"] != {"retryable_cleanup": 0, "durable_quarantine": 0}:
        _fail("raw report cleanup is not zero")
    if result["timing_ns"] > TIMEOUT_NS:
        _fail("raw report exceeds the 180 second timeout")
    if case["golden"]["status"] == "reviewed":
        for field in GOLDEN_AUDIT_KEYS:
            if execution[field] != case["golden"]["audit"][field]:
                _fail(f"raw report {field} does not match the reviewed G3 audit")


def _golden_matches(result: dict[str, Any], golden: dict[str, Any]) -> bool:
    if result["input_token_ids"] != _expand_input_spec(golden["input_token_spec"], "G3 golden input_token_spec"):
        return False
    for field in ("generated_token_ids", "visible_token_ids", "decode_input_token_ids", "output_text"):
        if golden[field] is not None and result[field] != golden[field]:
            return False
    return result["stop_reason"] == golden["stop_reason"]


def _observation_shape(target: str, observation: Any, phase: str, target_entry: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(observation, dict):
        _fail(f"{phase} AMD-SMI observation is missing")
    expected = {"selected_device", "health", "process"}
    _require_keys(observation, expected, f"{phase} observation")
    device = observation["selected_device"]
    _require_keys(device, {"target", "gpu_uuid", "gpu_bdf", "logical_device_index"}, f"{phase} selected device")
    if device != {"target": target, "gpu_uuid": target_entry["gpu_uuid"], "gpu_bdf": target_entry["gpu_bdf"], "logical_device_index": 0}:
        _fail(f"{phase} AMD-SMI selected-device identity is not the exact matrix device")
    health = observation["health"]
    _require_keys(health, {"available", "reliable", "state", "ras_uncorrectable_count"}, f"{phase} health")
    if health["available"] is not True or health["reliable"] is not True or health["state"] != "OK" or health["ras_uncorrectable_count"] != 0:
        _fail(f"{phase} AMD-SMI health is unavailable, unhealthy, or nonzero-RAS")
    process = observation["process"]
    _require_keys(process, {"available", "reliable", "state", "gpu_processes", "residual_runner_children"}, f"{phase} process")
    if process["available"] is not True or process["reliable"] is not True or process["state"] != "CLEAN":
        _fail(f"{phase} AMD-SMI process observation is unavailable or dirty")
    if process["gpu_processes"] != [] or process["residual_runner_children"] != []:
        _fail(f"{phase} GPU/process cleanup is not clean")
    return {
        "selected_device": dict(device),
        "health": dict(health),
        "process": dict(process),
    }


def _collect_observation(
    target: str,
    phase: str,
    target_entry: Mapping[str, Any],
    observer: Callable[[str, str], Any],
) -> dict[str, Any]:
    attempts = POST_OBSERVATION_ATTEMPTS if phase == "post" else 1
    for attempt in range(attempts):
        try:
            return _observation_shape(target, observer(target, phase), phase, target_entry)
        except G3Error:
            if attempt + 1 == attempts:
                raise
            time.sleep(POST_OBSERVATION_INTERVAL_SECONDS)
    raise AssertionError("unreachable")


def _child_process_ids(parent_pid: int) -> list[int]:
    parent_to_children: dict[int, list[int]] = {}
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return [-1]
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text(encoding="ascii").split()
            child_pid, parent = int(fields[0]), int(fields[3])
        except (OSError, ValueError, IndexError):
            continue
        parent_to_children.setdefault(parent, []).append(child_pid)
    result: list[int] = []
    pending = list(parent_to_children.get(parent_pid, []))
    while pending:
        child = pending.pop()
        result.append(child)
        pending.extend(parent_to_children.get(child, []))
    return sorted(set(result))


def _run_json_command(argv: list[str], *, timeout: int = 30) -> Any:
    try:
        completed = subprocess.run(argv, capture_output=True, check=False, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        _fail(f"AMD-SMI observation command failed: {exc}")
    if completed.returncode != 0 or completed.stderr != b"":
        _fail("AMD-SMI observation command did not exit cleanly")
    return _parse_json_bytes(completed.stdout, "AMD-SMI observation")


def _amd_smi_observation(target: str, phase: str) -> dict[str, Any]:
    """Read exactly the selected device; no device discovery fallback is allowed."""

    target_entry = TARGET_IDENTITIES[target]
    listed = _run_json_command([AMD_SMI_EXECUTABLE, "list", "-e", "--json"])
    if not isinstance(listed, list):
        _fail("AMD-SMI list is not an array")
    matches = [
        item for item in listed
        if isinstance(item, dict) and item.get("bdf", "").lower() == target_entry["gpu_bdf"]
        and item.get("hip_uuid") == target_entry["gpu_uuid"]
    ]
    if len(matches) != 1:
        _fail("AMD-SMI did not resolve exactly one selected canonical device")
    match = matches[0]
    if not _is_int(match.get("hip_id")) or match["hip_id"] < 0:
        _fail("AMD-SMI selected device has no physical index")
    metric = _run_json_command([AMD_SMI_EXECUTABLE, "metric", "-t", "-e", "--json", "-g", target_entry["gpu_bdf"]])
    if not isinstance(metric, dict) or not isinstance(metric.get("gpu_data"), list) or len(metric["gpu_data"]) != 1:
        _fail("AMD-SMI metric is not bound to one selected device")
    metric_record = metric["gpu_data"][0]
    ecc = metric_record.get("ecc") if isinstance(metric_record, dict) else None
    ras = ecc.get("total_uncorrectable_count") if isinstance(ecc, dict) else None
    if not _is_int(ras) or ras < 0:
        _fail("AMD-SMI health has no bounded RAS observation")
    process_doc = _run_json_command([AMD_SMI_EXECUTABLE, "process", "--json", "-g", target_entry["gpu_bdf"]])
    if not isinstance(process_doc, list) or len(process_doc) != 1 or not isinstance(process_doc[0], dict):
        _fail("AMD-SMI process observation is not bound to one selected device")
    process_list = process_doc[0].get("process_list")
    if process_list == [{"process_info": "No running processes detected"}]:
        gpu_processes: list[Any] = []
    elif isinstance(process_list, list):
        gpu_processes = [{"record_sha256": _sha256_bytes(_canonical_bytes(item))} for item in process_list]
    else:
        _fail("AMD-SMI process observation has no process list")
    children = _child_process_ids(os.getpid())
    return {
        "selected_device": {"target": target, **target_entry, "logical_device_index": 0},
        "health": {"available": True, "reliable": True, "state": "OK" if ras == 0 else "ERROR", "ras_uncorrectable_count": ras},
        "process": {"available": True, "reliable": True, "state": "CLEAN" if not gpu_processes and not children else "DIRTY", "gpu_processes": gpu_processes, "residual_runner_children": children},
    }


def _process_group_gone(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _signal_process_group(pid: int, sig: signal.Signals) -> bool:
    try:
        os.killpg(pid, sig)
        return True
    except ProcessLookupError:
        return False
    except OSError:
        return False


def _execute_subprocess(command: list[str], env: dict[str, str], cwd: Path | None, timeout_seconds: int) -> dict[str, Any]:
    started = time.monotonic_ns()
    try:
        process = subprocess.Popen(
            command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
        )
    except OSError as exc:
        _fail(f"cannot start G3 executable: {exc}")
    timed_out = False
    term_sent = False
    kill_sent = False
    stdout = b""
    stderr = b""
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        term_sent = _signal_process_group(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=TERMINATION_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            kill_sent = _signal_process_group(process.pid, signal.SIGKILL)
            try:
                stdout, stderr = process.communicate(timeout=TERMINATION_GRACE_SECONDS)
            except subprocess.TimeoutExpired as exc:
                _fail(f"G3 executable process group did not terminate after KILL: {exc}")
    duration_ns = time.monotonic_ns() - started
    if not isinstance(stdout, bytes):
        stdout = str(stdout).encode("utf-8")
    if not isinstance(stderr, bytes):
        stderr = str(stderr).encode("utf-8")
    if len(stdout) > MAX_REPORT_BYTES or len(stderr) > MAX_REPORT_BYTES:
        _fail("G3 executable output exceeds the bounded capture limit")
    return {
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": process.returncode,
        "timed_out": timed_out,
        "duration_ns": duration_ns,
        "term_sent": term_sent,
        "kill_sent": kill_sent,
        "process_group_gone": _process_group_gone(process.pid),
    }


def _capture_from_seam(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail("test command seam returned no capture")
    expected = {"stdout", "stderr", "exit_code", "timed_out", "duration_ns", "term_sent", "kill_sent", "process_group_gone"}
    _require_keys(value, expected, "command capture")
    if not isinstance(value["stdout"], bytes) or not isinstance(value["stderr"], bytes):
        _fail("command seam stdout/stderr must be bytes")
    if len(value["stdout"]) > MAX_REPORT_BYTES or len(value["stderr"]) > MAX_REPORT_BYTES:
        _fail("command seam output exceeds the bounded capture limit")
    if value["exit_code"] is not None and not _is_int(value["exit_code"]):
        _fail("command seam exit code is invalid")
    if not _is_int(value["duration_ns"]) or value["duration_ns"] < 0:
        _fail("command seam duration is invalid")
    for field in ("timed_out", "term_sent", "kill_sent", "process_group_gone"):
        if not isinstance(value[field], bool):
            _fail(f"command seam {field} is invalid")
    return dict(value)


def _write_bytes(path: Path, data: bytes, label: str) -> None:
    if path.exists() and (path.is_symlink() or not path.is_file()):
        _fail(f"{label} must be a regular non-symlink file: {path}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    except OSError as exc:
        _fail(f"cannot write {label} {path}: {exc}")


def _path_string(path: Path) -> str:
    return str(path)


def _expected_command(executable: Path, target: str, case_id: str) -> list[str]:
    if target not in TARGETS:
        _fail("G3 command target is outside the closed target set")
    if case_id == "g3-prompt-1":
        case_input = ["--prompt", "Hello"]
        max_new_tokens = 8
    elif case_id in {"g3-prompt-7", "g3-prompt-255", "g3-prompt-256", "g3-prompt-257"}:
        word_count = int(case_id.removeprefix("g3-prompt-"))
        case_input = ["--prompt", " ".join(["a"] * word_count)]
        max_new_tokens = 1 if word_count == 257 else 8
    elif case_id == "g3-unicode-chat-248046":
        case_input = ["--message", UNICODE_MESSAGE, "--thinking", "disabled"]
        max_new_tokens = 8
    else:
        _fail("G3 command case is outside the closed case set")
    return [
        str(executable), "generate",
        "--lock", str(MODEL_LOCK_PATH),
        "--cache", str(MODEL_CACHE_PATH),
        *case_input,
        "--max-new-tokens", str(max_new_tokens),
        "--device-index", "0",
        "--target", target,
        "--greedy",
    ]


def _new_run(run_id: str | None, attempt: int) -> dict[str, Any]:
    run_id = run_id or f"g3-run-{uuid.uuid4().hex}"
    if not isinstance(run_id, str) or RUN_ID_RE.fullmatch(run_id) is None:
        _fail("G3 run_id must be a unique g3-run hexadecimal identity")
    if not _is_int(attempt) or not 1 <= attempt <= 1_000_000:
        _fail("G3 run attempt must be between 1 and 1000000")
    identity = {"run_id": run_id, "attempt": attempt}
    return {**identity, "identity_sha256": _sha256_bytes(_canonical_bytes(identity))}


def _manifest_schema_check(manifest: dict[str, Any]) -> None:
    _schema_validate(manifest, REPORT_SCHEMA, "G3 evidence manifest")


def run_case(
    target: str,
    case_id: str,
    executable: Path,
    raw_report_path: Path,
    candidate: Mapping[str, str],
    *,
    matrix_path: Path = DEFAULT_MATRIX,
    run_id: str | None = None,
    attempt: int = 1,
    cwd: Path | None = ROOT,
    command_runner: Callable[[list[str], dict[str, str], Path | None, int], Any] | None = None,
    observation_provider: Callable[[str, str], Any] | None = None,
    test_only_fixture_matrix: bool = False,
) -> dict[str, Any]:
    """Execute exactly one target/case and return its closed evidence manifest.

    ``command_runner`` and ``observation_provider`` are injection seams for
    host-only tests.  The production CLI never exposes either seam and uses
    the external process-group timeout and AMD-SMI observer above.
    """

    matrix = validate_matrix(matrix_path, test_only_fixture_matrix=test_only_fixture_matrix)
    matrix_hash = _sha256_file(Path(matrix_path), "G3 matrix", MAX_MATRIX_BYTES)
    if target not in TARGETS:
        _fail("G3 run target is outside the closed target set")
    case = _expected_case(matrix, case_id)
    candidate_value = _candidate(candidate["commit"], candidate["tree"])
    if candidate_value["commit"] != matrix["candidate"]["commit"] or candidate_value["tree"] != matrix["candidate"]["tree"]:
        _fail("G3 run candidate is not the matrix-bound immutable candidate")
    run = _new_run(run_id, attempt)
    target_entry = _matrix_target_map(matrix)[target]
    executable = Path(executable)
    executable_hash = _sha256_file(executable, "G3 executable", MAX_BINARY_BYTES)
    if not os.access(executable, os.X_OK):
        _fail(f"G3 executable is not executable: {executable}")
    if executable_hash != target_entry["binary_sha256"]:
        _fail("G3 executable hash is not the exact matrix-bound artifact")
    device = {"target": target, "gpu_uuid": target_entry["gpu_uuid"], "gpu_bdf": target_entry["gpu_bdf"], "logical_device_index": 0}
    command = _expected_command(executable, target, case_id)
    if len(command) > MAX_COMMAND_ITEMS or any(len(item.encode("utf-8")) > MAX_COMMAND_ITEM_BYTES for item in command):
        _fail("G3 command exceeds the bounded command contract")
    environment = os.environ.copy()
    for name in VISIBILITY_NAMES:
        environment.pop(name, None)
    environment["ROCR_VISIBLE_DEVICES"] = device["gpu_uuid"]
    environment["SLLM_G3_TARGET"] = target
    environment["SLLM_G3_CASE_ID"] = case_id
    environment["SLLM_G3_CANDIDATE_COMMIT"] = candidate_value["commit"]
    environment["SLLM_G3_CANDIDATE_TREE"] = candidate_value["tree"]
    environment["SLLM_G3_MATRIX_SHA256"] = matrix_hash
    environment["SLLM_G3_RUN_ID"] = run["run_id"]
    environment["SLLM_G3_RUN_ATTEMPT"] = str(run["attempt"])

    observer = observation_provider or _amd_smi_observation
    pre = _collect_observation(target, "pre", target_entry, observer)
    capture: dict[str, Any] | None = None
    post: dict[str, Any]
    try:
        try:
            raw_capture = command_runner(command, environment, cwd, TIMEOUT_SECONDS) if command_runner else _execute_subprocess(command, environment, cwd, TIMEOUT_SECONDS)
            capture = _capture_from_seam(raw_capture) if command_runner else raw_capture
        except G3Error:
            raise
        except Exception as exc:
            _fail(f"G3 command seam failed: {exc}")
    finally:
        post = _collect_observation(target, "post", target_entry, observer)
    if capture is None:
        _fail("G3 run produced no command capture")

    _write_bytes(Path(raw_report_path), capture["stdout"], "raw report")
    raw_hash = _sha256_file(Path(raw_report_path), "raw report", MAX_REPORT_BYTES)
    executable_hash_after = _sha256_file(executable, "G3 executable after run", MAX_BINARY_BYTES)
    reasons: list[str] = []
    raw: dict[str, Any] | None = None
    try:
        raw, raw_hash_read, raw_bytes = _read_hashed_json(Path(raw_report_path), "raw generate report", MAX_REPORT_BYTES)
        if raw_hash_read != raw_hash:
            reasons.append("raw report changed while being read")
        result = _check_raw_report(raw, Path(raw_report_path), raw_bytes)
        _validate_raw_semantics(raw, result, case, target, matrix)
    except G3Error as exc:
        reasons.append(str(exc))
    if capture["exit_code"] != 0:
        reasons.append(f"executable exit was {capture['exit_code']}")
    if capture["timed_out"]:
        reasons.append("external timeout occurred")
    if capture["stderr"] != b"":
        reasons.append("stderr was not empty")
    if executable_hash_after != executable_hash:
        reasons.append("executable changed during the run")
    if capture["term_sent"] or capture["kill_sent"] or not capture["process_group_gone"]:
        reasons.append("external process cleanup was not a clean success")
    if pre["process"] != post["process"] or pre["selected_device"] != post["selected_device"] or pre["health"] != post["health"]:
        reasons.append("pre/post selected-device, health, or process observations differ")
    cleanup = {
        "pre_process_clean": pre["process"]["state"] == "CLEAN",
        "post_process_clean": post["process"]["state"] == "CLEAN",
        "process_group_gone": capture["process_group_gone"],
        "retryable_cleanup": 0 if capture["process_group_gone"] and post["process"]["state"] == "CLEAN" else 1,
        "durable_quarantine": 0,
    }
    manifest = {
        "schema_version": REPORT_VERSION,
        "record_kind": MANIFEST_KIND,
        "state": "PASS" if not reasons else "FAIL",
        "required": True,
        "failure_reason": "; ".join(reasons) if reasons else None,
        "run": run,
        "candidate": candidate_value,
        "matrix": {"path": _path_string(Path(matrix_path)), "matrix_id": matrix["matrix_id"], "sha256": matrix_hash},
        "target": {"target": target, "case_id": case_id, "backend": "hip"},
        "executable": {"path": _path_string(executable), "sha256": executable_hash},
        "raw_report": {"path": _path_string(Path(raw_report_path)), "sha256": raw_hash, "bytes": len(capture["stdout"])},
        "stderr": {"bytes": len(capture["stderr"]), "sha256": _sha256_bytes(capture["stderr"])},
        "command": command,
        "visibility": {"rocr_visible_devices": device["gpu_uuid"], "cleared": list(VISIBILITY_NAMES)},
        "observations": {"pre": pre, "post": post},
        "execution": {
            "exit_code": capture["exit_code"], "timed_out": capture["timed_out"],
            "timeout_seconds": TIMEOUT_SECONDS, "duration_ns": capture["duration_ns"],
            "term_sent": capture["term_sent"], "kill_sent": capture["kill_sent"],
            "process_group_gone": capture["process_group_gone"],
        },
        "cleanup": cleanup,
    }
    _manifest_schema_check(manifest)
    return manifest


def _matrix_path_from_manifest(manifest: dict[str, Any]) -> Path:
    path = Path(manifest["matrix"]["path"])
    return path if path.is_absolute() else ROOT / path


def _validate_manifest_semantics(manifest: dict[str, Any], matrix: dict[str, Any], matrix_hash: str, *, test_only_fixture_matrix: bool) -> None:
    if manifest["state"] != "PASS" or manifest["failure_reason"] is not None:
        _fail("G3 normalization refuses a failed run manifest")
    if manifest["required"] is not True:
        _fail("G3 manifest is not required evidence")
    candidate = _candidate_document(manifest["candidate"], "G3 manifest candidate")
    if candidate != manifest["candidate"] or candidate != {**matrix["candidate"], "candidate_sha256": candidate["candidate_sha256"]}:
        _fail("G3 manifest candidate/tree binding is stale or forged")
    if manifest["matrix"]["matrix_id"] != matrix["matrix_id"] or manifest["matrix"]["sha256"] != matrix_hash:
        _fail("G3 manifest matrix content binding is stale")
    if not test_only_fixture_matrix and matrix_hash != CANONICAL_MATRIX_SHA256:
        _fail("G3 manifest is not bound to the pinned canonical matrix")
    target = manifest["target"]["target"]
    case_id = manifest["target"]["case_id"]
    if target not in TARGETS or case_id not in CASE_IDS or manifest["target"]["backend"] != "hip":
        _fail("G3 manifest target/case is outside the closed matrix")
    run = manifest["run"]
    identity = {"run_id": run["run_id"], "attempt": run["attempt"]}
    if run["identity_sha256"] != _sha256_bytes(_canonical_bytes(identity)):
        _fail("G3 manifest run/attempt identity hash is stale")
    target_entry = _matrix_target_map(matrix)[target]
    expected_device = {"target": target, "gpu_uuid": target_entry["gpu_uuid"], "gpu_bdf": target_entry["gpu_bdf"], "logical_device_index": 0}
    for phase in ("pre", "post"):
        _observation_shape(target, manifest["observations"][phase], phase, target_entry)
        if manifest["observations"][phase]["selected_device"] != expected_device:
            _fail("G3 manifest selected-device identity is not matrix-bound")
    if manifest["observations"]["pre"] != manifest["observations"]["post"]:
        _fail("G3 manifest pre/post observations are not identical")
    executable = manifest["executable"]
    raw_report = manifest["raw_report"]
    _sha256(executable["sha256"], "G3 manifest executable hash")
    _sha256(raw_report["sha256"], "G3 manifest raw report hash")
    if not _is_int(raw_report["bytes"]) or raw_report["bytes"] > MAX_REPORT_BYTES:
        _fail("G3 manifest raw report size is unbounded")
    if manifest["stderr"] != {"bytes": 0, "sha256": EMPTY_SHA256}:
        _fail("G3 manifest does not prove empty stderr")
    if manifest["visibility"] != {"rocr_visible_devices": target_entry["gpu_uuid"], "cleared": list(VISIBILITY_NAMES)}:
        _fail("G3 manifest visibility isolation is incomplete")
    expected_command = _expected_command(Path(executable["path"]), target, case_id)
    if manifest["command"] != expected_command:
        _fail("G3 manifest invocation is not the exact target/case CLI contract")
    execution = manifest["execution"]
    if execution["exit_code"] != 0 or execution["timed_out"] is not False or execution["timeout_seconds"] != TIMEOUT_SECONDS:
        _fail("G3 manifest actual exit/timeout facts are not a successful 180-second run")
    if execution["term_sent"] or execution["kill_sent"] or execution["process_group_gone"] is not True:
        _fail("G3 manifest external timeout or process cleanup facts are unsafe")
    if manifest["cleanup"] != {
        "pre_process_clean": True, "post_process_clean": True,
        "process_group_gone": True, "retryable_cleanup": 0, "durable_quarantine": 0,
    }:
        _fail("G3 manifest cleanup facts are not fail-closed")


def normalize_manifest(manifest_path: Path, *, test_only_fixture_matrix: bool = False) -> dict[str, Any]:
    """Reopen a manifest's raw report and executable and derive one row."""

    manifest, manifest_hash, _ = _read_hashed_json(Path(manifest_path), "G3 evidence manifest", MAX_REPORT_BYTES)
    _manifest_schema_check(manifest)
    if manifest.get("record_kind") != MANIFEST_KIND:
        _fail("G3 normalize accepts evidence manifests only; standalone rows cannot yield PASS")
    matrix_path = _matrix_path_from_manifest(manifest)
    matrix = validate_matrix(matrix_path, test_only_fixture_matrix=test_only_fixture_matrix)
    matrix_hash = _sha256_file(matrix_path, "G3 matrix", MAX_MATRIX_BYTES)
    _validate_manifest_semantics(manifest, matrix, matrix_hash, test_only_fixture_matrix=test_only_fixture_matrix)

    target = manifest["target"]["target"]
    executable_path = Path(manifest["executable"]["path"])
    binary_hash_before = _sha256_file(executable_path, "G3 executable", MAX_BINARY_BYTES)
    target_entry = _matrix_target_map(matrix)[target]
    if binary_hash_before != manifest["executable"]["sha256"] or binary_hash_before != target_entry["binary_sha256"] or not os.access(executable_path, os.X_OK):
        _fail("G3 executable was modified, is not executable, or is not the manifest artifact")
    raw_path = Path(manifest["raw_report"]["path"])
    raw, raw_hash, raw_bytes = _read_hashed_json(raw_path, "raw generate report", MAX_REPORT_BYTES)
    if raw_hash != manifest["raw_report"]["sha256"] or len(raw_bytes) != manifest["raw_report"]["bytes"]:
        _fail("G3 raw report was modified after the run manifest")
    result = _check_raw_report(raw, raw_path, raw_bytes)
    case = _expected_case(matrix, manifest["target"]["case_id"])
    _validate_raw_semantics(raw, result, case, target, matrix)
    if result["execution"]["target"] != target:
        _fail("G3 raw report target is not manifest-bound")
    golden = case["golden"]
    if golden["status"] == "reviewed" and not _golden_matches({"input_token_ids": result["input_token_ids"], "generated_token_ids": result["generated_token_ids"], "visible_token_ids": result["visible_token_ids"], "decode_input_token_ids": result["decode_input_token_ids"], "output_text": result["output_text"], "stop_reason": result["stop_reason"]}, golden):
        _fail(f"raw report does not match reviewed golden for {case['id']}")
    device = manifest["observations"]["post"]["selected_device"]
    order = TARGETS.index(target) * len(CASE_IDS) + case["order"]
    execution = result["execution"]
    row = {
        "schema_version": REPORT_VERSION,
        "record_kind": ROW_KIND,
        "state": "PASS" if golden["status"] == "reviewed" else "PENDING",
        "golden_status": golden["status"],
        "row_id": f"generation-g3-{target}-{case['id']}",
        "order": order,
        "target": target,
        "case_id": case["id"],
        "run": dict(manifest["run"]),
        "candidate": dict(manifest["candidate"]),
        "matrix": {"matrix_id": matrix["matrix_id"], "sha256": matrix_hash},
        "model": dict(matrix["model"]),
        "device": dict(device),
        "binary_sha256": binary_hash_before,
        "raw_report_sha256": raw_hash,
        "input": {"kind": result["input_kind"], "token_ids": list(result["input_token_ids"]), "token_count": len(result["input_token_ids"])},
        "generation": {"generated_token_ids": list(result["generated_token_ids"]), "visible_token_ids": list(result["visible_token_ids"]), "decode_input_token_ids": list(result["decode_input_token_ids"]), "output_text": result["output_text"], "stop_reason": dict(result["stop_reason"])},
        "scope": dict(raw["scope"]),
        "audit": dict(execution),
        "cleanup": dict(manifest["cleanup"]),
        "execution": dict(manifest["execution"]),
    }
    binary_hash_after = _sha256_file(executable_path, "G3 executable after normalization", MAX_BINARY_BYTES)
    raw_hash_after = _sha256_file(raw_path, "raw report after normalization", MAX_REPORT_BYTES)
    if binary_hash_after != binary_hash_before or raw_hash_after != raw_hash:
        _fail("G3 executable or raw report changed during normalization")
    _assert_path_free(row)
    _schema_validate(row, REPORT_SCHEMA, "normalized G3 row")
    if manifest_hash == "0" * 64:
        _fail("G3 manifest hash is zero")
    return row


def test_only_normalize_fixture_manifest(manifest_path: Path) -> dict[str, Any]:
    """Explicit test-only fixture normalization entry point."""

    return normalize_manifest(manifest_path, test_only_fixture_matrix=True)


def _assert_path_free(value: Any, label: str = "normalized evidence") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"path", "paths", "command"}:
                _fail(f"{label} contains forbidden path/command material")
            _assert_path_free(item, label)
    elif isinstance(value, list):
        for item in value:
            _assert_path_free(item, label)


def aggregate_manifests(manifest_paths: Iterable[Path], *, test_only_fixture_matrix: bool = False) -> dict[str, Any]:
    """Reopen, rehash, and independently normalize exactly 12 manifests."""

    paths = tuple(Path(path) for path in manifest_paths)
    expected_count = len(TARGETS) * len(CASE_IDS)
    if len(paths) != expected_count:
        _fail(f"G3 aggregate requires exactly {expected_count} evidence manifests")
    if len({str(path.resolve()) for path in paths}) != len(paths):
        _fail("G3 aggregate contains duplicate manifest paths")
    rows = [normalize_manifest(path, test_only_fixture_matrix=test_only_fixture_matrix) for path in paths]
    first = rows[0]
    for row in rows:
        if row["state"] != "PASS" or row["golden_status"] != "reviewed":
            _fail("G3 aggregate refuses pending or non-PASS evidence")
        if row["candidate"] != first["candidate"] or row["matrix"] != first["matrix"] or row["model"] != first["model"]:
            _fail("G3 aggregate contains mixed candidate, matrix, or model evidence")
    by_id: dict[str, dict[str, Any]] = {}
    raw_hashes: set[str] = set()
    run_keys: set[tuple[str, int]] = set()
    for row in rows:
        if row["row_id"] in by_id:
            _fail("G3 aggregate contains duplicate evidence row")
        if row["raw_report_sha256"] in raw_hashes:
            _fail("G3 aggregate contains duplicate raw report evidence")
        run_key = (row["run"]["run_id"], row["run"]["attempt"])
        if run_key in run_keys:
            _fail("G3 aggregate contains duplicate run/attempt evidence")
        by_id[row["row_id"]] = row
        raw_hashes.add(row["raw_report_sha256"])
        run_keys.add(run_key)
    expected_ids = [f"generation-g3-{target}-{case_id}" for target in TARGETS for case_id in CASE_IDS]
    if set(by_id) != set(expected_ids):
        _fail("G3 aggregate has missing or unknown target/case evidence")
    ordered = [by_id[row_id] for row_id in expected_ids]
    matrix_path = _matrix_path_from_manifest(_read_json(paths[0], "G3 evidence manifest", MAX_REPORT_BYTES))
    matrix = validate_matrix(matrix_path, test_only_fixture_matrix=test_only_fixture_matrix)
    for case_id in CASE_IDS:
        pair = [row for row in ordered if row["case_id"] == case_id]
        if pair[0]["input"] != pair[1]["input"] or pair[0]["generation"] != pair[1]["generation"]:
            _fail(f"G3 targets disagree for case {case_id}")
    reviewed_cases = sum(case["golden"]["status"] == "reviewed" for case in matrix["cases"])
    audit_comparisons = 0
    for row in ordered:
        golden = _expected_case(matrix, row["case_id"])["golden"]
        if golden["status"] != "reviewed":
            _fail("G3 aggregate reviewed-golden audit set is incomplete")
        for key in GOLDEN_AUDIT_KEYS:
            audit_comparisons += 1
            if row["audit"][key] != golden["audit"][key]:
                _fail(f"G3 aggregate reviewed golden audit mismatch: {row['row_id']} {key}")
    aggregate = {
        "schema_version": AGGREGATE_VERSION,
        "aggregate_id": f"generation-g3-aggregate-{first['candidate']['candidate_sha256']}",
        "state": "PASS",
        "required": True,
        "candidate": dict(first["candidate"]),
        "matrix": dict(first["matrix"]),
        "model": dict(first["model"]),
        "rows": ordered,
        "counts": {
            "expected_rows": expected_count, "selected_rows": len(paths), "collected_rows": len(ordered),
            "passed_rows": len(ordered), "failed_rows": 0, "pending_rows": 0,
            "expected_cases": expected_count, "reviewed_cases": reviewed_cases,
            "reviewed_audit_comparisons": audit_comparisons, "reviewed_audit_mismatches": 0,
        },
        "scope": {"offline": True, "gpu_execution": True, "model_execution": True, "generation": True},
        "raw_data_policy": {"raw_model_bytes": False, "raw_trace": False, "report_contains_paths": False, "manifests_reopened": True},
    }
    _assert_path_free(aggregate)
    _schema_validate(aggregate, AGGREGATE_SCHEMA, "G3 aggregate")
    return aggregate


def test_only_aggregate_fixture_manifests(manifest_paths: Iterable[Path]) -> dict[str, Any]:
    """Explicit test-only fixture aggregate entry point."""

    return aggregate_manifests(manifest_paths, test_only_fixture_matrix=True)


# Deliberately no old row-taking implementation is retained.  These aliases
# keep descriptive callers readable while preserving the manifest-only gate.
normalize_raw_report = normalize_manifest
aggregate_reports = aggregate_manifests


def _write_json(path: Path, value: Any) -> None:
    if str(path) == "-":
        sys.stdout.buffer.write(_canonical_bytes(value))
        return
    _write_bytes(path, _canonical_bytes(value), "JSON output")


def _candidate_from_args(args: argparse.Namespace) -> dict[str, str]:
    if args.candidate is not None:
        document = _read_json(Path(args.candidate), "candidate", 16 * 1024)
        return _candidate_document(document, "candidate")
    if args.commit is None or args.tree is None:
        _fail("immutable candidate requires both --commit and --tree")
    return _candidate(args.commit, args.tree)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    validate = subparsers.add_parser("validate-matrix")
    validate.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    run = subparsers.add_parser("run")
    run.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    run.add_argument("--target", choices=TARGETS, required=True)
    run.add_argument("--case-id", choices=CASE_IDS, required=True)
    run.add_argument("--executable", type=Path, required=True)
    run.add_argument("--raw-report", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--attempt", type=int, required=True)
    run.add_argument("--candidate", type=Path)
    run.add_argument("--commit")
    run.add_argument("--tree")
    normalize = subparsers.add_parser("normalize")
    normalize.add_argument("--manifest", "--report", dest="manifest", type=Path, required=True)
    normalize.add_argument("--output", type=Path, required=True)
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--manifests", "--rows", dest="manifests", type=Path, nargs="+", required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.operation == "validate-matrix":
            validate_matrix(args.matrix)
            print("generation-g3 matrix: PASS")
            return 0
        if args.operation == "run":
            manifest = run_case(
                args.target, args.case_id, args.executable, args.raw_report,
                _candidate_from_args(args), matrix_path=args.matrix, run_id=args.run_id,
                attempt=args.attempt,
            )
            _write_json(args.output, manifest)
            print(f"generation-g3 run: {manifest['state']}")
            return 0 if manifest["state"] == "PASS" else 1
        if args.operation == "normalize":
            row = normalize_manifest(args.manifest)
            _write_json(args.output, row)
            print(f"generation-g3 normalize: {row['state']}")
            return 0 if row["state"] == "PASS" else 1
        aggregate = aggregate_manifests(args.manifests)
        _write_json(args.output, aggregate)
        print("generation-g3 aggregate: PASS")
        return 0
    except (G3Error, OSError, ValueError) as exc:
        print(f"generation-g3: FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
