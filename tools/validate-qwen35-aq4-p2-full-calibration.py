#!/usr/bin/env python3
"""Validate a full-vector Qwen3.5-9B source calibration artifact.

The validator is deliberately independent from the bounded v2 oracle
validator.  It streams the F32 sidecars in bounded chunks and refuses
symlinks, unstable files, duplicate JSON keys, unknown fields, bad offsets,
and identity drift from the parent sampled source oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import struct
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen35_aq4_p2_oracle as legacy_oracle  # noqa: E402


SCHEMA = "ullm.qwen35_aq4_source_calibration.v1"
ORACLE_KIND = "independent_source_full"
TARGET_SCHEMA = "ullm.qwen35_aq4_target_calibration.v1"
CASES_SCHEMA = "ullm.qwen35_aq4_source_calibration_cases.v1"
HIDDEN_SIZE = 4096
VOCAB_SIZE = 248320
TOP_K = 10
F32_BYTES = 4
ROW_BYTES = (HIDDEN_SIZE + VOCAB_SIZE) * F32_BYTES
MAX_CASE_FILE_BYTES = 4 * 1024 * 1024
MAX_CASES = 8192
MAX_ROWS = 16384
MAX_STEPS = 128
TARGET_MAX_CASE_FILE_BYTES = 64 * 1024 * 1024
TARGET_MAX_CASES = 24
TARGET_MAX_ROWS = 24
TARGET_MAX_STEPS = 1
MAX_TOPK = 32
MAX_CHUNK_ELEMENTS = 1_048_576
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_ROWS_FILE_BYTES = 64 * 1024 * 1024
MAX_ROW_LINE_BYTES = 64 * 1024
MAX_SHA_SUMS_BYTES = 8 * 1024 * 1024


class ValidationError(ValueError):
    pass


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_nonfinite(value: str) -> None:
    raise ValidationError(f"non-finite JSON number: {value}")


def _path_components(path: Path) -> list[Path]:
    absolute = path.absolute()
    return [Path(absolute.anchor), *(Path(absolute.anchor, *absolute.parts[1:index]) for index in range(1, len(absolute.parts) + 1))]


def no_symlink_components(path: Path, label: str, *, missing_leaf: bool = False) -> None:
    components = _path_components(path)
    for index, component in enumerate(components):
        try:
            info = os.lstat(component)
        except FileNotFoundError:
            if missing_leaf and index == len(components) - 1:
                return
            raise ValidationError(f"{label} path component is unavailable: {component}")
        if stat.S_ISLNK(info.st_mode):
            raise ValidationError(f"{label} path component is a symlink: {component}")


def stat_identity_info(info: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_nlink, info.st_mode


def regular(path: Path, label: str, *, max_bytes: int | None = None) -> Path:
    no_symlink_components(path, label)
    try:
        info = os.lstat(path)
    except OSError as error:
        raise ValidationError(f"{label} is unavailable: {error}") from error
    if not stat.S_ISREG(info.st_mode):
        raise ValidationError(f"{label} must be a regular file")
    if info.st_nlink != 1:
        raise ValidationError(f"{label} must have exactly one hard link")
    if max_bytes is not None and info.st_size > max_bytes:
        raise ValidationError(f"{label} exceeds {max_bytes} bytes")
    return path


def stat_identity(path: Path) -> tuple[int, int, int, int, int, int, int]:
    return stat_identity_info(os.lstat(path))


@contextmanager
def stable_fd(path: Path, label: str, *, max_bytes: int | None = None) -> Iterator[tuple[int, os.stat_result]]:
    regular(path, label, max_bytes=max_bytes)
    before = os.lstat(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as error:
        raise ValidationError(f"{label} cannot be opened safely: {error}") from error
    try:
        opened = os.fstat(fd)
        if stat_identity_info(opened) != stat_identity_info(before):
            raise ValidationError(f"{label} changed while being opened")
        yield fd, opened
        if stat_identity_info(os.fstat(fd)) != stat_identity_info(before):
            raise ValidationError(f"{label} changed while being read")
        no_symlink_components(path, label)
        if stat_identity(path) != stat_identity_info(before):
            raise ValidationError(f"{label} path changed while being read")
    finally:
        os.close(fd)


def read_fd_all(fd: int, max_bytes: int, label: str) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = os.read(fd, min(1024 * 1024, max_bytes + 1 - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > max_bytes:
            raise ValidationError(f"{label} exceeds {max_bytes} bytes")
    return b"".join(chunks)


def read_json(path: Path, label: str, *, max_bytes: int = MAX_CASE_FILE_BYTES) -> Any:
    with stable_fd(path, label, max_bytes=max_bytes) as (fd, _):
        raw = read_fd_all(fd, max_bytes, label)
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicate_keys, parse_constant=reject_nonfinite)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"invalid {label}: {error}") from error


def sha256_file(path: Path, label: str, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with stable_fd(path, label) as (fd, _):
        while chunk := os.read(fd, chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def ensure_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValidationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def exact_fields(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = set(value) if isinstance(value, dict) else set()
        raise ValidationError(f"{label} fields differ: missing={sorted(expected - actual)} extra={sorted(actual - expected)}")
    return value


def integer(value: Any, label: str, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or (maximum is not None and value > maximum):
        raise ValidationError(f"{label} is outside its integer bound")
    return value


def finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValidationError(f"{label} must be finite")
    return float(value)


def relative_path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValidationError(f"{label} must be a relative path")
    return legacy_oracle.safe_relative(root, value, label)


def load_cases(path: Path) -> tuple[list[dict[str, Any]], str]:
    raw = read_json(path, "calibration cases")
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "cases"} or raw["schema_version"] != CASES_SCHEMA:
        raise ValidationError("calibration cases schema differs")
    if not isinstance(raw["cases"], list) or not raw["cases"] or len(raw["cases"]) > MAX_CASES:
        raise ValidationError("calibration case count is outside bounds")
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    rows = 0
    for index, case in enumerate(raw["cases"]):
        required = {"case_id", "prompt_token_ids", "step_count"}
        allowed = required | {"semantic_input_id", "observation"}
        if not isinstance(case, dict) or not required <= set(case) or set(case) - allowed:
            raise ValidationError(f"cases[{index}] fields differ")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValidationError("case IDs must be unique non-empty strings")
        seen.add(case_id)
        tokens = case.get("prompt_token_ids")
        if not isinstance(tokens, list) or not tokens or len(tokens) > 4096:
            raise ValidationError(f"cases[{index}] token IDs are invalid")
        normalized_tokens = [integer(token, f"cases[{index}].prompt_token_ids", 0, VOCAB_SIZE - 1) for token in tokens]
        step_count = integer(case.get("step_count"), f"cases[{index}].step_count", 1, MAX_STEPS)
        rows += step_count
        if rows > MAX_ROWS:
            raise ValidationError("calibration row count exceeds bound")
        semantic = case.get("semantic_input_id", case_id)
        observation = case.get("observation", "first_token")
        if not isinstance(semantic, str) or not semantic or not isinstance(observation, str) or not observation:
            raise ValidationError(f"cases[{index}] semantic fields are invalid")
        cases.append({"case_id": case_id, "prompt_token_ids": normalized_tokens, "step_count": step_count, "semantic_input_id": semantic, "observation": observation})
    return cases, sha256_file(path, "calibration cases")


def read_f32_chunks(fd: int, offset: int, elements: int, chunk_elements: int) -> Iterator[list[float]]:
    remaining = elements
    cursor = offset
    while remaining:
        count = min(remaining, chunk_elements)
        raw = os.pread(fd, count * F32_BYTES, cursor)
        if len(raw) != count * F32_BYTES:
            raise ValidationError("vector sidecar ended before row boundary")
        yield list(struct.unpack(f"<{count}f", raw))
        cursor += len(raw)
        remaining -= count


def count_nonfinite(values: Iterator[list[float]]) -> int:
    count = 0
    for chunk in values:
        count += sum(1 for value in chunk if not math.isfinite(value))
    return count


def topk_from_chunks(values: Iterator[list[float]], elements: int, count: int) -> list[dict[str, Any]]:
    candidates: list[tuple[float, int]] = []
    token_id = 0
    for chunk in values:
        for value in chunk:
            if not math.isfinite(value):
                raise ValidationError("cannot rank non-finite logits")
            candidates.append((value, token_id))
            if len(candidates) > count * 2:
                candidates.sort(key=lambda item: (-item[0], item[1]))
                del candidates[count:]
            token_id += 1
    if token_id != elements:
        raise ValidationError("vector element count differs from row shape")
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [{"token_id": token, "logit": value} for value, token in candidates[:count]]


def topk_values_match(actual: list[dict[str, Any]], expected: list[dict[str, Any]]) -> bool:
    if len(actual) != len(expected):
        return False
    for left, right in zip(actual, expected):
        if left["token_id"] != right["token_id"]:
            return False
        try:
            left_f32 = struct.pack("<f", float(left["logit"]))
            right_f32 = struct.pack("<f", float(right["logit"]))
        except (TypeError, ValueError, OverflowError, struct.error):
            return False
        if left_f32 != right_f32:
            return False
    return True


def validate_topk(value: Any, label: str, count: int) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value or len(value) != count:
        raise ValidationError(f"{label} must contain exactly {count} entries")
    result: list[dict[str, Any]] = []
    previous: tuple[float, int] | None = None
    seen: set[int] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != {"token_id", "logit"}:
            raise ValidationError(f"{label}[{index}] fields differ")
        token_id = integer(item["token_id"], f"{label}[{index}].token_id", 0, VOCAB_SIZE - 1)
        logit = finite(item["logit"], f"{label}[{index}].logit")
        if token_id in seen:
            raise ValidationError(f"{label} contains a duplicate token")
        seen.add(token_id)
        key = (-logit, token_id)
        if previous is not None and key < previous:
            raise ValidationError(f"{label} is not tie-ordered")
        previous = key
        result.append({"token_id": token_id, "logit": logit})
    return result


def row_fields(row: Any, label: str) -> dict[str, Any]:
    expected = {"case_id", "step", "semantic_input_id", "observation", "input_token_ids_sha256", "hidden", "logits", "greedy_token_id", "topk", "finite"}
    if not isinstance(row, dict) or set(row) != expected:
        raise ValidationError(f"{label} fields differ")
    if not isinstance(row["case_id"], str) or not row["case_id"] or not isinstance(row["semantic_input_id"], str) or not row["semantic_input_id"] or not isinstance(row["observation"], str) or not row["observation"]:
        raise ValidationError(f"{label} identifiers are invalid")
    integer(row["step"], f"{label}.step", 0, MAX_STEPS - 1)
    ensure_sha(row["input_token_ids_sha256"], f"{label}.input_token_ids_sha256")
    for name, elements in (("hidden", HIDDEN_SIZE), ("logits", VOCAB_SIZE)):
        item = row[name]
        if not isinstance(item, dict) or set(item) != {"offset_bytes", "bytes", "elements", "dtype", "endianness", "sha256", "nonfinite_count"}:
            raise ValidationError(f"{label}.{name} fields differ")
        offset = integer(item["offset_bytes"], f"{label}.{name}.offset_bytes")
        byte_count = integer(item["bytes"], f"{label}.{name}.bytes", 1)
        if byte_count != elements * F32_BYTES or integer(item["elements"], f"{label}.{name}.elements", 1) != elements or item["dtype"] != "f32" or item["endianness"] != "little":
            raise ValidationError(f"{label}.{name} shape contract differs")
        if offset % F32_BYTES:
            raise ValidationError(f"{label}.{name}.offset_bytes is not aligned")
        ensure_sha(item["sha256"], f"{label}.{name}.sha256")
        integer(item["nonfinite_count"], f"{label}.{name}.nonfinite_count")
    if not isinstance(row["finite"], bool):
        raise ValidationError(f"{label}.finite must be boolean")
    vector_finite = row["hidden"]["nonfinite_count"] == 0 and row["logits"]["nonfinite_count"] == 0
    if vector_finite:
        integer(row["greedy_token_id"], f"{label}.greedy_token_id", 0, VOCAB_SIZE - 1)
        validate_topk(row["topk"], f"{label}.topk", TOP_K)
        if row["topk"][0]["token_id"] != row["greedy_token_id"] or row["finite"] is not True:
            raise ValidationError(f"{label} greedy/finite fields differ")
    elif row["greedy_token_id"] is not None or row["topk"] is not None or row["finite"] is not False:
        raise ValidationError(f"{label} blocked rows must use null greedy/top-k")
    return row


def read_rows(path: Path, parser: Any = row_fields) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    with stable_fd(path, "rows file", max_bytes=MAX_ROWS_FILE_BYTES) as (fd, _):
        pending = b""
        line_number = 0
        while True:
            chunk = os.read(fd, 64 * 1024)
            if not chunk:
                break
            pending += chunk
            if len(pending) > MAX_ROW_LINE_BYTES and b"\n" not in pending:
                raise ValidationError("rows line exceeds byte bound")
            while b"\n" in pending:
                raw, pending = pending.split(b"\n", 1)
                line_number += 1
                if not raw or len(raw) > MAX_ROW_LINE_BYTES:
                    raise ValidationError(f"rows line {line_number} is empty or oversized")
                try:
                    value = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicate_keys, parse_constant=reject_nonfinite)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise ValidationError(f"invalid rows JSON at line {line_number}: {error}") from error
                row = parser(value, f"rows[{line_number}]")
                key = (row["case_id"], row["step"])
                if key in rows:
                    raise ValidationError("rows contain duplicate case/step")
                rows[key] = row
                if len(rows) > MAX_ROWS:
                    raise ValidationError("rows exceed row-count bound")
        if pending:
            line_number += 1
            if len(pending) > MAX_ROW_LINE_BYTES:
                raise ValidationError("rows final line exceeds byte bound")
            try:
                value = json.loads(pending.decode("utf-8"), object_pairs_hook=reject_duplicate_keys, parse_constant=reject_nonfinite)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValidationError(f"invalid rows JSON at line {line_number}: {error}") from error
            row = parser(value, f"rows[{line_number}]")
            key = (row["case_id"], row["step"])
            if key in rows:
                raise ValidationError("rows contain duplicate case/step")
            rows[key] = row
        if not rows:
            raise ValidationError("rows file is empty")
    return rows


def verify_sha_sums(root: Path, expected_files: set[str]) -> None:
    sums_path = root / "SHA256SUMS"
    entries: dict[str, str] = {}
    with stable_fd(sums_path, "SHA256SUMS", max_bytes=MAX_SHA_SUMS_BYTES) as (fd, _):
        raw = read_fd_all(fd, MAX_SHA_SUMS_BYTES, "SHA256SUMS")
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise ValidationError(f"SHA256SUMS must be ASCII: {error}") from error
    for line_number, line in enumerate(lines, 1):
        fields = line.split("  ", 1)
        if len(fields) != 2:
            raise ValidationError(f"SHA256SUMS line {line_number} is invalid")
        digest, name = fields
        ensure_sha(digest, f"SHA256SUMS line {line_number}")
        if name in entries:
            raise ValidationError("SHA256SUMS contains duplicate path")
        entries[name] = digest
    if set(entries) != expected_files:
        raise ValidationError("SHA256SUMS file set differs")
    for name, expected in entries.items():
        path = relative_path(root, name, f"SHA256SUMS {name}")
        if sha256_file(path, name) != expected:
            raise ValidationError(f"SHA256SUMS digest differs for {name}")


def artifact_inventory(root: Path, expected_files: set[str]) -> None:
    expected_dirs = {""}
    for name in expected_files:
        parent = Path(name).parent
        while parent.as_posix() not in {".", ""}:
            expected_dirs.add(parent.as_posix())
            parent = parent.parent
    actual_files: set[str] = set()
    actual_dirs: set[str] = {""}
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        base = Path(current)
        for name in list(directories):
            path = base / name
            rel = path.relative_to(root).as_posix()
            info = os.lstat(path)
            if stat.S_ISLNK(info.st_mode):
                raise ValidationError(f"artifact contains symlink: {rel}")
            if not stat.S_ISDIR(info.st_mode):
                raise ValidationError(f"artifact contains non-directory: {rel}")
            actual_dirs.add(rel)
        for name in files:
            path = base / name
            rel = path.relative_to(root).as_posix()
            info = os.lstat(path)
            if stat.S_ISLNK(info.st_mode):
                raise ValidationError(f"artifact contains symlink: {rel}")
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValidationError(f"artifact file is not single-link regular: {rel}")
            actual_files.add(rel)
    if actual_files != expected_files or actual_dirs != expected_dirs:
        raise ValidationError(f"artifact exact file set differs: files={sorted(actual_files ^ expected_files)} dirs={sorted(actual_dirs ^ expected_dirs)}")


def legacy_check(root: Path, manifest: dict[str, Any], rows: dict[tuple[str, int], dict[str, Any]], hidden_path: Path, logits_path: Path) -> dict[str, Any]:
    parent = manifest["parent_sampled_oracle"]
    legacy_manifest_path = Path(parent["path"])
    if parent["schema_version"] == SCHEMA:
        parent_root = legacy_manifest_path.parent
        parent_report = validate(parent_root)
        if parent_report["manifest_sha256"] != parent["manifest_sha256"]:
            raise ValidationError("direct source calibration parent manifest hash differs")
        parent_manifest = read_json(legacy_manifest_path, "direct source calibration parent manifest")
        for key in ("model_id", "model_revision"):
            if parent_manifest["identity"].get(key) != manifest["identity"].get(key):
                raise ValidationError(f"direct source calibration parent identity differs: {key}")
        if parent_manifest["identity"]["source_checkpoint"]["aggregate_sha256"] != manifest["identity"]["source_checkpoint"]["aggregate_sha256"] or parent_manifest["identity"]["tokenizer"]["aggregate_sha256"] != manifest["identity"]["tokenizer"]["aggregate_sha256"]:
            raise ValidationError("direct source calibration parent checkpoint/tokenizer differs")
        expected = {
            "status": "not_applicable",
            "legacy_manifest_sha256": parent["manifest_sha256"],
            "legacy_payload_sha256": "",
            "row_count": manifest["cases"]["row_count"],
            "hidden_sample_max_abs_diff": 0.0,
            "logit_sample_max_abs_diff": 0.0,
        }
        if manifest["legacy_cross_check"] != expected:
            raise ValidationError("direct source calibration parent summary differs")
        return expected
    legacy = legacy_oracle.validate_manifest(legacy_manifest_path.parent, expected_kind="independent_source")
    if sha256_file(legacy_manifest_path, "parent sampled manifest") != parent["manifest_sha256"]:
        raise ValidationError("parent sampled manifest hash differs")
    for key in ("model_id", "model_revision"):
        if legacy["identity"].get(key) != manifest["identity"].get(key):
            raise ValidationError(f"parent sampled identity differs: {key}")
    if legacy["identity"]["source_checkpoint"]["aggregate_sha256"] != manifest["identity"]["source_checkpoint"]["aggregate_sha256"] or legacy["identity"]["tokenizer"]["aggregate_sha256"] != manifest["identity"]["tokenizer"]["aggregate_sha256"]:
        raise ValidationError("parent sampled checkpoint/tokenizer differs")
    summary = manifest["legacy_cross_check"]
    old_fields = {"status", "legacy_manifest_sha256", "legacy_payload_sha256", "row_count", "hidden_sample_max_abs_diff", "logit_sample_max_abs_diff"}
    new_fields = old_fields | {"split_manifest_path", "policy_path", "calibration_cases_path", "split_manifest_sha256", "policy_sha256", "calibration_cases_sha256", "excluded_case_ids", "excluded_case_ids_sha256", "overlap_case_ids"}
    if set(summary) not in (old_fields, new_fields):
        raise ValidationError("legacy cross-check summary fields differ")
    legacy_records = list(legacy_oracle.payload_records(legacy_manifest_path.parent, legacy))
    overlap = {(old["case_id"], old["step"]) for old in legacy_records if (old["case_id"], old["step"]) in rows}
    if set(summary) == new_fields:
        if summary["legacy_manifest_sha256"] != sha256_file(legacy_manifest_path, "parent sampled manifest") or summary["legacy_payload_sha256"] != legacy["payload"]["sha256"] or not all(isinstance(summary[field], str) and Path(summary[field]).is_absolute() for field in ("split_manifest_path", "policy_path", "calibration_cases_path")):
            raise ValidationError("legacy parent/path hash binding differs")
        split_path = Path(summary["split_manifest_path"]); policy_path = Path(summary["policy_path"]); calibration_cases_path = Path(summary["calibration_cases_path"])
        if sha256_file(split_path, "split manifest") != summary["split_manifest_sha256"] or sha256_file(policy_path, "policy") != summary["policy_sha256"] or sha256_file(calibration_cases_path, "calibration cases") != summary["calibration_cases_sha256"]:
            raise ValidationError("legacy split/policy/calibration hash binding differs")
        split_value = read_json(split_path, "split manifest")
        exclusions = split_value.get("attempt2_exclusions") if isinstance(split_value, dict) else None
        if not isinstance(exclusions, dict) or exclusions.get("case_ids") != summary["excluded_case_ids"]:
            raise ValidationError("legacy split exclusion binding differs")
        encoded_exclusions = json.dumps(summary["excluded_case_ids"], ensure_ascii=True, separators=(",", ":")).encode("utf-8")
        if summary["excluded_case_ids_sha256"] != hashlib.sha256(encoded_exclusions).hexdigest() or summary["overlap_case_ids"] != sorted({case_id for case_id, _step in overlap}):
            raise ValidationError("legacy exclusion/overlap binding differs")
        if summary["status"] == "not_applicable_disjoint_by_policy":
            if overlap or not {old["case_id"] for old in legacy_records}.issubset(set(summary["excluded_case_ids"])) or summary["row_count"] != 0 or summary["hidden_sample_max_abs_diff"] != 0.0 or summary["logit_sample_max_abs_diff"] != 0.0:
                raise ValidationError("legacy disjoint policy summary differs")
            return summary
    checked = 0
    hidden_max = 0.0
    logits_max = 0.0
    with stable_fd(hidden_path, "hidden sidecar") as (hidden, _), stable_fd(logits_path, "logits sidecar") as (logit, _):
        for old in legacy_records:
            row = rows.get((old["case_id"], old["step"]))
            if row is None:
                raise ValidationError(f"parent sampled overlapping row is missing: {old['case_id']}/{old['step']}")
            hidden_values = next(read_f32_chunks(hidden, row["hidden"]["offset_bytes"], HIDDEN_SIZE, 65536))
            # Hidden is only 4096 values, so one bounded chunk is expected.
            if len(hidden_values) != HIDDEN_SIZE:
                raise ValidationError("parent hidden row length differs")
            for index, expected in zip(old["hidden_sample"]["indices"], old["hidden_sample"]["values"]):
                hidden_max = max(hidden_max, abs(hidden_values[index] - float(expected)))
            for index, expected in zip(old["logit_sample"]["indices"], old["logit_sample"]["values"]):
                raw = os.pread(logit, F32_BYTES, row["logits"]["offset_bytes"] + index * F32_BYTES)
                if len(raw) != F32_BYTES:
                    raise ValidationError("parent logit sample is truncated")
                logits_max = max(logits_max, abs(struct.unpack("<f", raw)[0] - float(expected)))
            if row["greedy_token_id"] != old["greedy_token_id"] or row["topk"] != old["topk"]:
                raise ValidationError(f"parent sampled top-k/greedy differs: {old['case_id']}/{old['step']}")
            checked += 1
    expected = {"status": "passed", "legacy_manifest_sha256": sha256_file(legacy_manifest_path, "parent sampled manifest"), "legacy_payload_sha256": legacy["payload"]["sha256"], "row_count": checked, "hidden_sample_max_abs_diff": hidden_max, "logit_sample_max_abs_diff": logits_max}
    if set(summary) == new_fields:
        expected = {**expected, "split_manifest_path": summary["split_manifest_path"], "policy_path": summary["policy_path"], "calibration_cases_path": summary["calibration_cases_path"], "split_manifest_sha256": summary["split_manifest_sha256"], "policy_sha256": summary["policy_sha256"], "calibration_cases_sha256": summary["calibration_cases_sha256"], "excluded_case_ids": summary["excluded_case_ids"], "excluded_case_ids_sha256": summary["excluded_case_ids_sha256"], "overlap_case_ids": summary["overlap_case_ids"]}
    if summary != expected:
        raise ValidationError("legacy cross-check summary differs")
    return summary


ROOT_FIELDS = {"schema_version", "oracle_kind", "status", "evidence_class", "usable_as_source_evidence", "promotion_eligible", "created_utc", "identity", "parent_sampled_oracle", "vector_contract", "limits", "cases", "files", "runtime", "legacy_cross_check"}


def _file_records(value: Any, label: str) -> None:
    if not isinstance(value, list):
        raise ValidationError(f"{label} must be a list")
    for index, item in enumerate(value):
        exact_fields(item, {"file", "bytes", "sha256"}, f"{label}[{index}]")
        if not isinstance(item["file"], str) or not item["file"]:
            raise ValidationError(f"{label}[{index}].file is invalid")
        integer(item["bytes"], f"{label}[{index}].bytes")
        ensure_sha(item["sha256"], f"{label}[{index}].sha256")


def validate_manifest_shape(manifest: Any, schemas: set[str]) -> dict[str, Any]:
    manifest = exact_fields(manifest, ROOT_FIELDS, "manifest")
    if manifest["schema_version"] not in schemas:
        raise ValidationError("calibration manifest schema differs")
    allowed_kinds = {"independent_source_full", "aq4_target", "same_artifact_all_m1", "aq4_optimized"}
    if manifest["oracle_kind"] not in allowed_kinds:
        raise ValidationError("calibration oracle kind differs")
    if manifest["status"] not in {"available", "blocked"} or manifest["evidence_class"] not in {"production", "blocked", "synthetic_fixture"} or not isinstance(manifest["usable_as_source_evidence"], bool) or manifest["promotion_eligible"] is not False:
        raise ValidationError("calibration status/evidence fields differ")
    legacy_oracle.validate_utc(manifest["created_utc"])
    base_identity = {"artifact", "model_id", "model_revision", "source_checkpoint", "tokenizer", "hidden_size", "vocab_size"}
    target_extra = {"package_content_sha256", "package_manifest_sha256", "worker_binary_sha256"}
    identity_keys = set(manifest["identity"]) if isinstance(manifest["identity"], dict) else set()
    expected_identity = base_identity if manifest["schema_version"] == SCHEMA else base_identity | target_extra
    if identity_keys != expected_identity:
        raise ValidationError("identity fields differ")
    identity = manifest["identity"]
    artifact = exact_fields(identity["artifact"], {"package_manifest_sha256", "artifact_manifest_sha256"}, "identity.artifact")
    for key, value in artifact.items():
        if value is not None:
            ensure_sha(value, f"identity.artifact.{key}")
    checkpoint = exact_fields(identity["source_checkpoint"], {"aggregate_sha256", "dtype", "files", "root"}, "identity.source_checkpoint")
    tokenizer = exact_fields(identity["tokenizer"], {"aggregate_sha256", "files", "root"}, "identity.tokenizer")
    if identity["model_id"] != "Qwen/Qwen3.5-9B" or not isinstance(identity["model_revision"], str) or identity["hidden_size"] != HIDDEN_SIZE or identity["vocab_size"] != VOCAB_SIZE:
        raise ValidationError("calibration model/vector identity differs")
    ensure_sha(checkpoint["aggregate_sha256"], "source checkpoint aggregate")
    ensure_sha(tokenizer["aggregate_sha256"], "tokenizer aggregate")
    _file_records(checkpoint["files"], "identity.source_checkpoint.files")
    _file_records(tokenizer["files"], "identity.tokenizer.files")
    if not isinstance(checkpoint["dtype"], str) or not isinstance(checkpoint["root"], str) or not isinstance(tokenizer["root"], str):
        raise ValidationError("checkpoint/tokenizer metadata differs")
    for key in target_extra & identity_keys:
        ensure_sha(identity[key], f"identity.{key}")
    parent = exact_fields(manifest["parent_sampled_oracle"], {"path", "manifest_sha256", "schema_version"}, "parent_sampled_oracle")
    allowed_parent_schemas = {legacy_oracle.SOURCE_SCHEMA}
    if manifest["schema_version"] == TARGET_SCHEMA:
        allowed_parent_schemas.add(SCHEMA)
    if not isinstance(parent["path"], str) or not Path(parent["path"]).is_absolute() or parent["schema_version"] not in allowed_parent_schemas:
        raise ValidationError("parent sampled oracle binding differs")
    ensure_sha(parent["manifest_sha256"], "parent sampled oracle manifest")
    contract = exact_fields(manifest["vector_contract"], {"hidden_shape", "logits_shape", "dtype", "endianness", "layout", "chunk_elements", "row_bytes", "semantic_hidden", "semantic_logits"}, "vector_contract")
    if contract["hidden_shape"] != [HIDDEN_SIZE] or contract["logits_shape"] != [VOCAB_SIZE] or contract["dtype"] != "f32" or contract["endianness"] != "little" or contract["layout"] != "flat" or integer(contract["chunk_elements"], "chunk_elements", 1, MAX_CHUNK_ELEMENTS) <= 0 or contract["row_bytes"] != ROW_BYTES or not isinstance(contract["semantic_hidden"], str) or not isinstance(contract["semantic_logits"], str):
        raise ValidationError("vector contract differs")
    limits = exact_fields(manifest["limits"], {"max_case_file_bytes", "max_cases", "max_rows", "max_steps"}, "limits")
    if manifest["schema_version"] == TARGET_SCHEMA:
        expected_limits = {"max_case_file_bytes": TARGET_MAX_CASE_FILE_BYTES, "max_cases": TARGET_MAX_CASES, "max_rows": TARGET_MAX_ROWS, "max_steps": TARGET_MAX_STEPS}
        if limits != expected_limits:
            raise ValidationError("target limits differ")
    else:
        for field, upper in (("max_case_file_bytes", MAX_CASE_FILE_BYTES), ("max_cases", MAX_CASES), ("max_rows", MAX_ROWS), ("max_steps", MAX_STEPS)):
            integer(limits[field], f"limits.{field}", 1, upper)
    cases = exact_fields(manifest["cases"], {"path", "sha256", "case_count", "row_count"}, "cases")
    if not isinstance(cases["path"], str) or not Path(cases["path"]).is_absolute():
        raise ValidationError("cases.path must be absolute")
    ensure_sha(cases["sha256"], "cases.sha256")
    integer(cases["case_count"], "cases.case_count", 1, MAX_CASES)
    integer(cases["row_count"], "cases.row_count", 1, MAX_ROWS)
    exact_fields(manifest["files"], {"rows", "hidden", "logits"}, "files")
    runtime = exact_fields(manifest["runtime"], {"runtime", "transformers", "torch", "safetensors", "python", "device", "dtype", "low_cpu_mem_usage", "torch_num_threads", "torch_num_interop_threads", "model_loads", "inference_mode", "full_vocab_ranking", "max_resident_logit_rows", "memory_preflight", "disk_preflight", "run"}, "runtime")
    exact_fields(runtime["memory_preflight"], {"checkpoint_bytes", "mem_total_bytes", "mem_available_bytes", "required_headroom_bytes", "headroom_factor", "status"}, "runtime.memory_preflight")
    exact_fields(runtime["disk_preflight"], {"expected_vector_bytes", "required_free_bytes", "free_bytes", "status"}, "runtime.disk_preflight")
    exact_fields(runtime["run"], {"row_count", "nonfinite_rows", "elapsed_seconds"}, "runtime.run")
    legacy_fields = manifest["legacy_cross_check"]
    if not isinstance(legacy_fields, dict) or set(legacy_fields) not in ({"status", "legacy_manifest_sha256", "legacy_payload_sha256", "row_count", "hidden_sample_max_abs_diff", "logit_sample_max_abs_diff"}, {"status", "legacy_manifest_sha256", "legacy_payload_sha256", "split_manifest_path", "policy_path", "calibration_cases_path", "split_manifest_sha256", "policy_sha256", "calibration_cases_sha256", "excluded_case_ids", "excluded_case_ids_sha256", "overlap_case_ids", "row_count", "hidden_sample_max_abs_diff", "logit_sample_max_abs_diff"}):
        raise ValidationError("legacy_cross_check fields differ")
    return manifest


def validate(root: Path) -> dict[str, Any]:
    no_symlink_components(root, "artifact root")
    root_info = os.lstat(root)
    if not stat.S_ISDIR(root_info.st_mode):
        raise ValidationError("artifact root must be a real directory")
    manifest = validate_manifest_shape(read_json(root / "manifest.json", "calibration manifest", max_bytes=MAX_MANIFEST_BYTES), {SCHEMA, TARGET_SCHEMA})
    if manifest["oracle_kind"] not in {ORACLE_KIND, "aq4_target"}:
        raise ValidationError("calibration oracle kind differs")
    files = manifest["files"]
    rows_path = relative_path(root, files["rows"], "rows file")
    hidden_path = relative_path(root, files["hidden"], "hidden file")
    logits_path = relative_path(root, files["logits"], "logits file")
    expected_artifact_files = {"manifest.json", "SHA256SUMS", files["rows"], files["hidden"], files["logits"]}
    artifact_inventory(root, expected_artifact_files)
    cases_path = Path(manifest["cases"]["path"])
    cases, cases_sha = load_cases(cases_path)
    if manifest["cases"]["sha256"] != cases_sha or manifest["cases"]["case_count"] != len(cases):
        raise ValidationError("calibration cases binding differs")
    rows = read_rows(rows_path)
    expected_keys = {(case["case_id"], step) for case in cases for step in range(case["step_count"])}
    if manifest["cases"]["row_count"] != len(rows):
        raise ValidationError("calibration cases row binding differs")
    if manifest["status"] == "available" and set(rows) != expected_keys:
        raise ValidationError("rows case/step coverage differs")
    if manifest["status"] == "blocked" and not set(rows) <= expected_keys:
        raise ValidationError("blocked rows exceed case/step coverage")
    previous_hidden = previous_logits = 0
    nonfinite_rows = 0
    chunk_elements = manifest["vector_contract"]["chunk_elements"]
    expected_hidden_bytes = len(rows) * HIDDEN_SIZE * F32_BYTES
    expected_logits_bytes = len(rows) * VOCAB_SIZE * F32_BYTES
    with stable_fd(hidden_path, "hidden sidecar", max_bytes=expected_hidden_bytes) as (hidden, hidden_info), stable_fd(logits_path, "logits sidecar", max_bytes=expected_logits_bytes) as (logits, logits_info):
        if hidden_info.st_size != expected_hidden_bytes or logits_info.st_size != expected_logits_bytes:
            raise ValidationError("vector sidecar size differs from row count")
        for key in sorted(rows):
            row = rows[key]
            if row["hidden"]["offset_bytes"] != previous_hidden or row["logits"]["offset_bytes"] != previous_logits:
                raise ValidationError("vector offsets are not contiguous")
            for name, fd, elements, item in (("hidden", hidden, HIDDEN_SIZE, row["hidden"]), ("logits", logits, VOCAB_SIZE, row["logits"])):
                digest = hashlib.sha256()
                nonfinite = value_count = 0
                for chunk in read_f32_chunks(fd, item["offset_bytes"], elements, chunk_elements):
                    encoded = struct.pack(f"<{len(chunk)}f", *chunk)
                    digest.update(encoded)
                    nonfinite += sum(1 for value in chunk if not math.isfinite(value))
                    value_count += len(chunk)
                if digest.hexdigest() != item["sha256"] or value_count != elements or nonfinite != item["nonfinite_count"]:
                    raise ValidationError(f"{name} row hash/nonfinite differs for {key}")
                if name == "hidden":
                    previous_hidden += item["bytes"]
                else:
                    previous_logits += item["bytes"]
            row_nonfinite = row["hidden"]["nonfinite_count"] + row["logits"]["nonfinite_count"] > 0
            nonfinite_rows += int(row_nonfinite)
            if not row_nonfinite:
                ranked = topk_from_chunks(read_f32_chunks(logits, row["logits"]["offset_bytes"], VOCAB_SIZE, chunk_elements), VOCAB_SIZE, TOP_K)
                if not topk_values_match(ranked, row["topk"]):
                    raise ValidationError(f"top-k ranking differs for {key}")
    blocked = nonfinite_rows > 0
    expected_usable_as_source = manifest["schema_version"] == SCHEMA and not blocked
    if (manifest["status"] == "blocked") != blocked or (manifest["evidence_class"] == "blocked") != blocked or manifest["usable_as_source_evidence"] != expected_usable_as_source or manifest["runtime"]["run"]["nonfinite_rows"] != nonfinite_rows:
        raise ValidationError("manifest blocked status differs from vector finiteness")
    if blocked:
        parent_check = manifest["legacy_cross_check"]
    else:
        parent_check = legacy_check(root, manifest, rows, hidden_path, logits_path)
        if manifest["legacy_cross_check"] != parent_check:
            raise ValidationError("legacy cross-check summary differs")
    verify_sha_sums(root, expected_artifact_files - {"SHA256SUMS"})
    return {"schema_version": manifest["schema_version"], "status": "blocked" if blocked else "valid", "artifact_root": str(root.resolve()), "manifest_sha256": sha256_file(root / "manifest.json", "manifest"), "row_count": len(rows), "nonfinite_rows": nonfinite_rows, "legacy_cross_check": parent_check}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(validate(args.artifact), ensure_ascii=True, sort_keys=True))
        return 0
    except (ValidationError, legacy_oracle.OracleError, OSError, ValueError) as error:
        print(f"Qwen3.5 full source calibration validation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
