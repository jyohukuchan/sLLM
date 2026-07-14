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
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen35_aq4_p2_oracle as legacy_oracle  # noqa: E402


SCHEMA = "ullm.qwen35_aq4_source_calibration.v1"
ORACLE_KIND = "independent_source_full"
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
MAX_TOPK = 32
MAX_CHUNK_ELEMENTS = 1_048_576


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


def regular(path: Path, label: str, *, max_bytes: int | None = None) -> Path:
    if path.is_symlink():
        raise ValidationError(f"{label} must not be a symlink")
    try:
        info = path.lstat()
    except OSError as error:
        raise ValidationError(f"{label} is unavailable: {error}") from error
    if not stat.S_ISREG(info.st_mode):
        raise ValidationError(f"{label} must be a regular file")
    if max_bytes is not None and info.st_size > max_bytes:
        raise ValidationError(f"{label} exceeds {max_bytes} bytes")
    return path


def stat_identity(path: Path) -> tuple[int, int, int, int, int]:
    info = path.lstat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_mode


def read_json(path: Path, label: str, *, max_bytes: int = MAX_CASE_FILE_BYTES) -> Any:
    regular(path, label, max_bytes=max_bytes)
    before = stat_identity(path)
    raw = path.read_bytes()
    after = stat_identity(path)
    if before != after:
        raise ValidationError(f"{label} changed while being read")
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicate_keys, parse_constant=reject_nonfinite)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"invalid {label}: {error}") from error


def sha256_file(path: Path, label: str, chunk_bytes: int = 1024 * 1024) -> str:
    regular(path, label)
    before = stat_identity(path)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_bytes):
            digest.update(chunk)
    after = stat_identity(path)
    if before != after:
        raise ValidationError(f"{label} changed while being hashed")
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
        if not isinstance(case, dict) or set(case) - {"case_id", "prompt_token_ids", "step_count", "semantic_input_id", "observation"}:
            raise ValidationError(f"cases[{index}] contains unknown fields")
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
        cases.append({"case_id": case_id, "prompt_token_ids": normalized_tokens, "step_count": step_count, "semantic_input_id": case.get("semantic_input_id", case_id), "observation": case.get("observation", "first_token")})
    return cases, sha256_file(path, "calibration cases")


def read_f32_chunks(handle: Any, offset: int, elements: int, chunk_elements: int) -> Iterator[list[float]]:
    handle.seek(offset)
    remaining = elements
    while remaining:
        count = min(remaining, chunk_elements)
        raw = handle.read(count * F32_BYTES)
        if len(raw) != count * F32_BYTES:
            raise ValidationError("vector sidecar ended before row boundary")
        yield list(struct.unpack(f"<{count}f", raw))
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
    integer(row["greedy_token_id"], f"{label}.greedy_token_id", 0, VOCAB_SIZE - 1)
    validate_topk(row["topk"], f"{label}.topk", TOP_K)
    if row["topk"][0]["token_id"] != row["greedy_token_id"] or not isinstance(row["finite"], bool):
        raise ValidationError(f"{label} greedy/finite fields differ")
    return row


def verify_sha_sums(root: Path, expected_files: set[str]) -> None:
    sums_path = regular(root / "SHA256SUMS", "SHA256SUMS", max_bytes=8 * 1024 * 1024)
    entries: dict[str, str] = {}
    before = stat_identity(sums_path)
    for line_number, line in enumerate(sums_path.read_text(encoding="ascii").splitlines(), 1):
        fields = line.split("  ", 1)
        if len(fields) != 2:
            raise ValidationError(f"SHA256SUMS line {line_number} is invalid")
        digest, name = fields
        ensure_sha(digest, f"SHA256SUMS line {line_number}")
        if name in entries:
            raise ValidationError("SHA256SUMS contains duplicate path")
        entries[name] = digest
    if before != stat_identity(sums_path) or entries.keys() != expected_files:
        raise ValidationError("SHA256SUMS file set differs")
    for name, expected in entries.items():
        path = relative_path(root, name, f"SHA256SUMS {name}")
        if sha256_file(path, name) != expected:
            raise ValidationError(f"SHA256SUMS digest differs for {name}")


def legacy_check(root: Path, manifest: dict[str, Any], rows: dict[tuple[str, int], dict[str, Any]], hidden_path: Path, logits_path: Path) -> dict[str, Any]:
    parent = manifest["parent_sampled_oracle"]
    legacy_manifest_path = Path(parent["path"])
    legacy = legacy_oracle.validate_manifest(legacy_manifest_path.parent, expected_kind="independent_source")
    if sha256_file(legacy_manifest_path, "parent sampled manifest") != parent["manifest_sha256"]:
        raise ValidationError("parent sampled manifest hash differs")
    for key in ("model_id", "model_revision"):
        if legacy["identity"].get(key) != manifest["identity"].get(key):
            raise ValidationError(f"parent sampled identity differs: {key}")
    if legacy["identity"]["source_checkpoint"]["aggregate_sha256"] != manifest["identity"]["source_checkpoint"]["aggregate_sha256"] or legacy["identity"]["tokenizer"]["aggregate_sha256"] != manifest["identity"]["tokenizer"]["aggregate_sha256"]:
        raise ValidationError("parent sampled checkpoint/tokenizer differs")
    checked = 0
    hidden_max = 0.0
    logits_max = 0.0
    with hidden_path.open("rb") as hidden, logits_path.open("rb") as logit:
        for old in legacy_oracle.payload_records(legacy_manifest_path.parent, legacy):
            row = rows.get((old["case_id"], old["step"]))
            if row is None:
                raise ValidationError(f"parent sampled row is missing: {old['case_id']}/{old['step']}")
            hidden_values = next(read_f32_chunks(hidden, row["hidden"]["offset_bytes"], HIDDEN_SIZE, 65536))
            # Hidden is only 4096 values, so one bounded chunk is expected.
            if len(hidden_values) != HIDDEN_SIZE:
                raise ValidationError("parent hidden row length differs")
            for index, expected in zip(old["hidden_sample"]["indices"], old["hidden_sample"]["values"]):
                hidden_max = max(hidden_max, abs(hidden_values[index] - float(expected)))
            logit_values: list[float] = []
            for chunk in read_f32_chunks(logit, row["logits"]["offset_bytes"], VOCAB_SIZE, 65536):
                logit_values.extend(chunk)
            for index, expected in zip(old["logit_sample"]["indices"], old["logit_sample"]["values"]):
                logits_max = max(logits_max, abs(logit_values[index] - float(expected)))
            if row["greedy_token_id"] != old["greedy_token_id"] or row["topk"] != old["topk"]:
                raise ValidationError(f"parent sampled top-k/greedy differs: {old['case_id']}/{old['step']}")
            checked += 1
    return {"status": "passed", "legacy_manifest_sha256": sha256_file(legacy_manifest_path, "parent sampled manifest"), "legacy_payload_sha256": legacy["payload"]["sha256"], "row_count": checked, "hidden_sample_max_abs_diff": hidden_max, "logit_sample_max_abs_diff": logits_max}


def validate(root: Path) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise ValidationError("artifact root must be a real directory")
    manifest = read_json(root / "manifest.json", "calibration manifest", max_bytes=16 * 1024 * 1024)
    expected = {"schema_version", "oracle_kind", "status", "evidence_class", "usable_as_source_evidence", "promotion_eligible", "created_utc", "identity", "parent_sampled_oracle", "vector_contract", "limits", "cases", "files", "runtime", "legacy_cross_check"}
    if not isinstance(manifest, dict) or set(manifest) != expected or manifest["schema_version"] != SCHEMA or manifest["oracle_kind"] != ORACLE_KIND:
        raise ValidationError("calibration manifest fields/schema differ")
    if manifest["status"] not in {"available", "blocked"} or manifest["evidence_class"] not in {"production", "blocked", "synthetic_fixture"} or not isinstance(manifest["usable_as_source_evidence"], bool) or manifest["promotion_eligible"] is not False:
        raise ValidationError("calibration status/evidence fields differ")
    legacy_oracle.validate_utc(manifest["created_utc"])
    identity = exact_fields(manifest["identity"], {"artifact", "model_id", "model_revision", "source_checkpoint", "tokenizer", "hidden_size", "vocab_size"}, "identity")
    exact_fields(identity["artifact"], {"package_manifest_sha256", "artifact_manifest_sha256"}, "identity.artifact")
    exact_fields(identity["source_checkpoint"], {"aggregate_sha256", "dtype", "files", "root"}, "identity.source_checkpoint")
    exact_fields(identity["tokenizer"], {"aggregate_sha256", "files", "root"}, "identity.tokenizer")
    if identity.get("model_id") != "Qwen/Qwen3.5-9B" or identity.get("hidden_size") != HIDDEN_SIZE or identity.get("vocab_size") != VOCAB_SIZE:
        raise ValidationError("calibration model/vector identity differs")
    ensure_sha(identity["source_checkpoint"]["aggregate_sha256"], "source checkpoint aggregate")
    ensure_sha(identity["tokenizer"]["aggregate_sha256"], "tokenizer aggregate")
    contract = exact_fields(manifest["vector_contract"], {"hidden_shape", "logits_shape", "dtype", "endianness", "layout", "chunk_elements", "row_bytes", "semantic_hidden", "semantic_logits"}, "vector_contract")
    if not isinstance(contract, dict) or contract.get("hidden_shape") != [HIDDEN_SIZE] or contract.get("logits_shape") != [VOCAB_SIZE] or contract.get("dtype") != "f32" or contract.get("endianness") != "little" or contract.get("layout") != "flat" or integer(contract.get("chunk_elements"), "chunk_elements", 1, MAX_CHUNK_ELEMENTS) <= 0 or contract.get("row_bytes") != ROW_BYTES:
        raise ValidationError("vector contract differs")
    limits = exact_fields(manifest["limits"], {"max_case_file_bytes", "max_cases", "max_rows", "max_steps"}, "limits")
    for field, upper in (("max_case_file_bytes", MAX_CASE_FILE_BYTES), ("max_cases", MAX_CASES), ("max_rows", MAX_ROWS), ("max_steps", MAX_STEPS)):
        if integer(limits.get(field), f"limits.{field}", 1, upper) <= 0:
            raise ValidationError(f"limits.{field} is invalid")
    files = exact_fields(manifest["files"], {"rows", "hidden", "logits"}, "files")
    if not isinstance(files, dict) or set(files) != {"rows", "hidden", "logits"}:
        raise ValidationError("calibration file map differs")
    rows_path = relative_path(root, files["rows"], "rows file")
    hidden_path = relative_path(root, files["hidden"], "hidden file")
    logits_path = relative_path(root, files["logits"], "logits file")
    cases_path = Path(manifest["cases"].get("path", ""))
    cases, cases_sha = load_cases(cases_path)
    if manifest["cases"].get("sha256") != cases_sha or manifest["cases"].get("case_count") != len(cases) or manifest["cases"].get("row_count") != sum(case["step_count"] for case in cases):
        raise ValidationError("calibration cases binding differs")
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    with rows_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                row = row_fields(json.loads(line, object_pairs_hook=reject_duplicate_keys, parse_constant=reject_nonfinite), f"rows[{line_number}]")
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValidationError(f"invalid rows JSON at line {line_number}: {error}") from error
            key = (row["case_id"], row["step"])
            if key in rows:
                raise ValidationError("rows contain duplicate case/step")
            rows[key] = row
    expected_keys = {(case["case_id"], step) for case in cases for step in range(case["step_count"])}
    if set(rows) != expected_keys:
        raise ValidationError("rows case/step coverage differs")
    hidden_size = hidden_path.stat().st_size
    logits_size = logits_path.stat().st_size
    if hidden_size != len(rows) * HIDDEN_SIZE * F32_BYTES or logits_size != len(rows) * VOCAB_SIZE * F32_BYTES:
        raise ValidationError("vector sidecar size differs from row count")
    previous_hidden = previous_logits = 0
    nonfinite_rows = 0
    chunk_elements = contract["chunk_elements"]
    with hidden_path.open("rb") as hidden, logits_path.open("rb") as logits:
        for key in sorted(rows):
            row = rows[key]
            if row["hidden"]["offset_bytes"] != previous_hidden or row["logits"]["offset_bytes"] != previous_logits:
                raise ValidationError("vector offsets are not contiguous")
            for name, handle, elements, item in (("hidden", hidden, HIDDEN_SIZE, row["hidden"]), ("logits", logits, VOCAB_SIZE, row["logits"])):
                digest = hashlib.sha256()
                finite_count = 0
                value_count = 0
                for chunk in read_f32_chunks(handle, item["offset_bytes"], elements, chunk_elements):
                    encoded = struct.pack(f"<{len(chunk)}f", *chunk)
                    digest.update(encoded)
                    finite_count += sum(1 for value in chunk if not math.isfinite(value))
                    value_count += len(chunk)
                if digest.hexdigest() != item["sha256"] or value_count != elements or finite_count != item["nonfinite_count"]:
                    raise ValidationError(f"{name} row hash/nonfinite differs for {key}")
                if name == "hidden":
                    previous_hidden += item["bytes"]
                else:
                    previous_logits += item["bytes"]
            if row["hidden"]["nonfinite_count"] or row["logits"]["nonfinite_count"]:
                nonfinite_rows += 1
            # Ranking validation is a second bounded pass over the logit row.
            ranked = topk_from_chunks(read_f32_chunks(logits, row["logits"]["offset_bytes"], VOCAB_SIZE, chunk_elements), VOCAB_SIZE, TOP_K)
            if ranked != row["topk"]:
                raise ValidationError(f"top-k ranking differs for {key}")
    parent_check = legacy_check(root, manifest, rows, hidden_path, logits_path)
    exact_fields(manifest["legacy_cross_check"], {"status", "legacy_manifest_sha256", "legacy_payload_sha256", "row_count", "hidden_sample_max_abs_diff", "logit_sample_max_abs_diff"}, "legacy_cross_check")
    if manifest["legacy_cross_check"] != parent_check:
        raise ValidationError("legacy cross-check summary differs")
    expected_sidecars = {files["rows"], files["hidden"], files["logits"], "manifest.json"}
    actual_files = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() and not path.is_symlink()}
    if "SHA256SUMS" not in actual_files:
        raise ValidationError("SHA256SUMS is missing")
    verify_sha_sums(root, actual_files - {"SHA256SUMS"})
    return {"schema_version": SCHEMA, "status": "valid" if nonfinite_rows == 0 else "blocked", "artifact_root": str(root.resolve()), "manifest_sha256": sha256_file(root / "manifest.json", "manifest"), "row_count": len(rows), "nonfinite_rows": nonfinite_rows, "legacy_cross_check": parent_check}


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
