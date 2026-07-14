#!/usr/bin/env python3
"""Compare full-vector Qwen3.5 calibration sidecars without retaining rows.

``source_gate`` compares an independent BF16 source artifact with an AQ4
target sidecar.  ``path_gate`` compares AQ4 all-M=1 with an optimized AQ4
sidecar.  The two identities are intentionally checked separately and this
tool never creates or modifies a threshold policy from observed values.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
import stat
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator

import importlib.util


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = Path(__file__).with_name("validate-qwen35-aq4-p2-full-calibration.py")
_VALIDATOR_SPEC = importlib.util.spec_from_file_location("qwen35_full_calibration_validator", VALIDATOR_PATH)
if _VALIDATOR_SPEC is None or _VALIDATOR_SPEC.loader is None:
    raise RuntimeError("full calibration validator is unavailable")
_VALIDATOR = importlib.util.module_from_spec(_VALIDATOR_SPEC)
_VALIDATOR_SPEC.loader.exec_module(_VALIDATOR)


SCHEMA = "ullm.qwen35_aq4_calibration_comparison.v1"
SOURCE_SCHEMA = "ullm.qwen35_aq4_source_calibration.v1"
TARGET_SCHEMA = "ullm.qwen35_aq4_target_calibration.v1"
F32_BYTES = 4
HIDDEN_SIZE = 4096
VOCAB_SIZE = 248320
TOP_K = 10
MAX_CHUNK_ELEMENTS = 1_048_576
MAX_ROWS = 16384


class ComparisonError(ValueError):
    pass


def verify_loaded(artifact: dict[str, Any]) -> None:
    for name, path in artifact["tracked"].items():
        try:
            if _VALIDATOR.stat_identity(path) != artifact["fingerprints"][name]:
                raise ComparisonError(f"artifact changed after validation: {name}")
        except (OSError, _VALIDATOR.ValidationError) as error:
            raise ComparisonError(f"artifact became unavailable after validation: {name}: {error}") from error


def publish_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise ComparisonError("renameat2(RENAME_NOREPLACE) is unavailable")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise ComparisonError(f"refusing to overwrite existing output: {destination}")
        raise ComparisonError(f"exclusive output publish failed: {os.strerror(error)}")
    parent_fd = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ComparisonError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_nonfinite(value: str) -> None:
    raise ComparisonError(f"non-finite JSON number: {value}")


def regular(path: Path, label: str, max_bytes: int | None = None) -> Path:
    try:
        _VALIDATOR.regular(path, label, max_bytes=max_bytes)
    except _VALIDATOR.ValidationError as error:
        raise ComparisonError(str(error)) from error
    return path


def stat_identity(path: Path) -> tuple[int, int, int, int, int, int, int]:
    return _VALIDATOR.stat_identity(path)


def read_json(path: Path, label: str, max_bytes: int = 16 * 1024 * 1024) -> Any:
    try:
        return _VALIDATOR.read_json(path, label, max_bytes=max_bytes)
    except _VALIDATOR.ValidationError as error:
        raise ComparisonError(str(error)) from error


def sha256_file(path: Path, label: str) -> str:
    try:
        return _VALIDATOR.sha256_file(path, label)
    except _VALIDATOR.ValidationError as error:
        raise ComparisonError(str(error)) from error


def ensure_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ComparisonError(f"{label} must be a lowercase SHA-256 digest")
    return value


def relative_path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ComparisonError(f"{label} must be relative")
    return _VALIDATOR.relative_path(root, value, label)


def read_chunks(fd: int, offset: int, elements: int, chunk_elements: int) -> Iterator[list[float]]:
    remaining = elements
    cursor = offset
    while remaining:
        count = min(remaining, chunk_elements)
        raw = os.pread(fd, count * F32_BYTES, cursor)
        if len(raw) != count * F32_BYTES:
            raise ComparisonError("vector sidecar ended before row boundary")
        yield list(struct.unpack(f"<{count}f", raw))
        cursor += len(raw)
        remaining -= count


def finite_metrics(reference: Iterator[list[float]], candidate: Iterator[list[float]], elements: int, chunk_elements: int | None = None) -> dict[str, Any]:
    ref_sq = 0.0
    delta_sq = 0.0
    max_abs = 0.0
    ref_nonfinite = 0
    candidate_nonfinite = 0
    seen = 0
    ref_chunk: list[float] = []
    candidate_chunk: list[float] = []
    ref_index = candidate_index = 0
    while seen < elements:
        if ref_index == len(ref_chunk):
            try:
                ref_chunk = next(reference)
            except StopIteration as error:
                raise ComparisonError("reference vector stream ended early") from error
            ref_index = 0
            if not ref_chunk:
                raise ComparisonError("reference vector stream yielded an empty chunk")
        if candidate_index == len(candidate_chunk):
            try:
                candidate_chunk = next(candidate)
            except StopIteration as error:
                raise ComparisonError("candidate vector stream ended early") from error
            candidate_index = 0
            if not candidate_chunk:
                raise ComparisonError("candidate vector stream yielded an empty chunk")
        count = min(len(ref_chunk) - ref_index, len(candidate_chunk) - candidate_index, elements - seen)
        for left, right in zip(ref_chunk[ref_index:ref_index + count], candidate_chunk[candidate_index:candidate_index + count]):
            if not math.isfinite(left):
                ref_nonfinite += 1
            if not math.isfinite(right):
                candidate_nonfinite += 1
            if math.isfinite(left) and math.isfinite(right):
                difference = float(right) - float(left)
                max_abs = max(max_abs, abs(difference))
                delta_sq += difference * difference
                ref_sq += float(left) * float(left)
        ref_index += count
        candidate_index += count
        seen += count
    if ref_index != len(ref_chunk) or candidate_index != len(candidate_chunk):
        raise ComparisonError("vector stream contains surplus elements")
    try:
        next(reference)
        raise ComparisonError("reference vector stream contains surplus chunks")
    except StopIteration:
        pass
    try:
        next(candidate)
        raise ComparisonError("candidate vector stream contains surplus chunks")
    except StopIteration:
        pass
    if ref_nonfinite or candidate_nonfinite:
        return {"relative_l2": None, "max_abs": None, "reference_nonfinite_count": ref_nonfinite, "candidate_nonfinite_count": candidate_nonfinite}
    return {"relative_l2": math.sqrt(delta_sq) / max(math.sqrt(ref_sq), 1e-30), "max_abs": max_abs, "reference_nonfinite_count": 0, "candidate_nonfinite_count": 0}


def topk_from_chunks(values: Iterator[list[float]], elements: int, count: int) -> list[dict[str, Any]]:
    candidates: list[tuple[float, int]] = []
    token_id = 0
    for chunk in values:
        for value in chunk:
            if not math.isfinite(value):
                raise ComparisonError("top-k ranking encountered a non-finite logit")
            candidates.append((value, token_id))
            if len(candidates) > count * 2:
                candidates.sort(key=lambda item: (-item[0], item[1]))
                del candidates[count:]
            token_id += 1
    if token_id != elements:
        raise ComparisonError("top-k stream length differs from vocabulary")
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [{"token_id": token, "logit": value} for value, token in candidates[:count]]


def parse_row(value: Any, label: str) -> dict[str, Any]:
    try:
        return _VALIDATOR.row_fields(value, label)
    except _VALIDATOR.ValidationError as error:
        raise ComparisonError(str(error)) from error


def load_artifact(root: Path) -> dict[str, Any]:
    try:
        _VALIDATOR.no_symlink_components(root, "artifact root")
        if not stat.S_ISDIR(os.lstat(root).st_mode):
            raise ComparisonError("artifact root must be a real directory")
        manifest = _VALIDATOR.validate_manifest_shape(read_json(root / "manifest.json", "artifact manifest"), {SOURCE_SCHEMA, TARGET_SCHEMA})
    except _VALIDATOR.ValidationError as error:
        raise ComparisonError(str(error)) from error
    contract = manifest["vector_contract"]
    chunk_elements = contract["chunk_elements"]
    files = manifest["files"]
    try:
        cases, cases_sha = _VALIDATOR.load_cases(Path(manifest["cases"]["path"]))
    except _VALIDATOR.ValidationError as error:
        raise ComparisonError(str(error)) from error
    if cases_sha != manifest["cases"]["sha256"] or len(cases) != manifest["cases"]["case_count"]:
        raise ComparisonError("artifact cases binding differs")
    rows_path = relative_path(root, files["rows"], "artifact rows")
    hidden_path = relative_path(root, files["hidden"], "artifact hidden")
    logits_path = relative_path(root, files["logits"], "artifact logits")
    expected_files = {"manifest.json", "SHA256SUMS", files["rows"], files["hidden"], files["logits"]}
    try:
        _VALIDATOR.artifact_inventory(root, expected_files)
        rows = _VALIDATOR.read_rows(rows_path, parse_row)
    except _VALIDATOR.ValidationError as error:
        raise ComparisonError(str(error)) from error
    if len(rows) != manifest["cases"]["row_count"]:
        raise ComparisonError("artifact cases row binding differs")
    previous_hidden = previous_logits = 0
    nonfinite_rows = 0
    expected_hidden = len(rows) * HIDDEN_SIZE * F32_BYTES
    expected_logits = len(rows) * VOCAB_SIZE * F32_BYTES
    try:
        with _VALIDATOR.stable_fd(hidden_path, "artifact hidden", max_bytes=expected_hidden) as (hidden, hidden_info), _VALIDATOR.stable_fd(logits_path, "artifact logits", max_bytes=expected_logits) as (logit, logits_info):
            if hidden_info.st_size != expected_hidden or logits_info.st_size != expected_logits:
                raise ComparisonError("artifact sidecar size differs from row count")
            for key in sorted(rows):
                row = rows[key]
                if row["hidden"]["offset_bytes"] != previous_hidden or row["logits"]["offset_bytes"] != previous_logits:
                    raise ComparisonError("artifact vector offsets are not contiguous")
                for name, handle, elements, item in (("hidden", hidden, HIDDEN_SIZE, row["hidden"]), ("logits", logit, VOCAB_SIZE, row["logits"])):
                    digest = hashlib.sha256()
                    nonfinite = 0
                    for chunk in read_chunks(handle, item["offset_bytes"], elements, chunk_elements):
                        digest.update(struct.pack(f"<{len(chunk)}f", *chunk))
                        nonfinite += sum(1 for value in chunk if not math.isfinite(value))
                    if digest.hexdigest() != item["sha256"] or nonfinite != item.get("nonfinite_count"):
                        raise ComparisonError(f"artifact {name} row hash/nonfinite differs for {key}")
                    if name == "hidden":
                        previous_hidden += item["bytes"]
                    else:
                        previous_logits += item["bytes"]
                row_nonfinite_count = row["hidden"]["nonfinite_count"] + row["logits"]["nonfinite_count"]
                if row_nonfinite_count == 0:
                    ranked = topk_from_chunks(read_chunks(logit, row["logits"]["offset_bytes"], VOCAB_SIZE, chunk_elements), VOCAB_SIZE, TOP_K)
                    if ranked != row["topk"]:
                        raise ComparisonError(f"artifact top-k differs for {key}")
                nonfinite_rows += int(row_nonfinite_count > 0)
    except _VALIDATOR.ValidationError as error:
        raise ComparisonError(str(error)) from error
    blocked = nonfinite_rows > 0
    if (manifest["status"] == "blocked") != blocked or manifest["runtime"]["run"]["nonfinite_rows"] != nonfinite_rows:
        raise ComparisonError("artifact blocked status differs from vector finiteness")
    try:
        _VALIDATOR.verify_sha_sums(root, expected_files - {"SHA256SUMS"})
    except _VALIDATOR.ValidationError as error:
        raise ComparisonError(str(error)) from error
    tracked = {name: root / name for name in expected_files}
    fingerprints = {name: _VALIDATOR.stat_identity(path) for name, path in tracked.items()}
    return {"root": root, "manifest": manifest, "manifest_sha256": sha256_file(root / "manifest.json", "artifact manifest"), "rows": rows, "hidden": hidden_path, "logits": logits_path, "chunk_elements": chunk_elements, "tracked": tracked, "fingerprints": fingerprints, "nonfinite_rows": nonfinite_rows}


def check_identity(reference: dict[str, Any], candidate: dict[str, Any], compare_kind: str) -> None:
    left = reference["manifest"]["identity"]
    right = candidate["manifest"]["identity"]
    if left.get("model_id") != right.get("model_id") or left.get("model_revision") != right.get("model_revision") or left["tokenizer"]["aggregate_sha256"] != right["tokenizer"]["aggregate_sha256"]:
        raise ComparisonError("reference/candidate model revision or tokenizer differs")
    if compare_kind == "path_gate":
        for field in ("package_content_sha256", "package_manifest_sha256", "worker_binary_sha256"):
            if left.get(field) != right.get(field):
                raise ComparisonError(f"path oracle identity differs: {field}")


def compare(reference: dict[str, Any], candidate: dict[str, Any], compare_kind: str, output: Path) -> dict[str, Any]:
    verify_loaded(reference)
    verify_loaded(candidate)
    if compare_kind == "source_gate" and reference["manifest"].get("oracle_kind") != "independent_source_full":
        raise ComparisonError("source_gate reference must be independent_source_full")
    if compare_kind == "source_gate" and candidate["manifest"].get("oracle_kind") != "aq4_target":
        raise ComparisonError("source_gate candidate must be aq4_target")
    if compare_kind == "path_gate" and reference["manifest"].get("oracle_kind") not in {"same_artifact_all_m1", "aq4_target"}:
        raise ComparisonError("path_gate reference must be AQ4 all-M1")
    if compare_kind == "path_gate" and candidate["manifest"].get("oracle_kind") != "aq4_optimized":
        raise ComparisonError("path_gate candidate must be aq4_optimized")
    check_identity(reference, candidate, compare_kind)
    if set(reference["rows"]) != set(candidate["rows"]):
        raise ComparisonError("reference/candidate row coverage differs")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        _VALIDATOR.no_symlink_components(output.parent, "comparison output parent")
        _VALIDATOR.no_symlink_components(output, "comparison output", missing_leaf=True)
    except _VALIDATOR.ValidationError as error:
        raise ComparisonError(str(error)) from error
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.incomplete-", dir=output.parent))
    try:
        result_rows = temporary / "rows.jsonl"
        row_count = 0
        nonfinite_rows = 0
        greedy_mismatch = 0
        overlap_min = TOP_K
        max_hidden_relative = 0.0
        max_hidden_abs = 0.0
        max_logits_relative = 0.0
        max_logits_abs = 0.0
        with _VALIDATOR.stable_fd(reference["hidden"], "reference hidden") as (reference_hidden, _), _VALIDATOR.stable_fd(candidate["hidden"], "candidate hidden") as (candidate_hidden, _), _VALIDATOR.stable_fd(reference["logits"], "reference logits") as (reference_logits, _), _VALIDATOR.stable_fd(candidate["logits"], "candidate logits") as (candidate_logits, _), result_rows.open("w", encoding="utf-8") as rows_out:
            for key in sorted(reference["rows"]):
                left = reference["rows"][key]
                right = candidate["rows"][key]
                if left["input_token_ids_sha256"] != right["input_token_ids_sha256"]:
                    raise ComparisonError(f"input token identity differs for {key}")
                hidden = finite_metrics(read_chunks(reference_hidden, left["hidden"]["offset_bytes"], HIDDEN_SIZE, reference["chunk_elements"]), read_chunks(candidate_hidden, right["hidden"]["offset_bytes"], HIDDEN_SIZE, candidate["chunk_elements"]), HIDDEN_SIZE, min(reference["chunk_elements"], candidate["chunk_elements"]))
                logits = finite_metrics(read_chunks(reference_logits, left["logits"]["offset_bytes"], VOCAB_SIZE, reference["chunk_elements"]), read_chunks(candidate_logits, right["logits"]["offset_bytes"], VOCAB_SIZE, candidate["chunk_elements"]), VOCAB_SIZE, min(reference["chunk_elements"], candidate["chunk_elements"]))
                if hidden["relative_l2"] is not None:
                    max_hidden_relative = max(max_hidden_relative, hidden["relative_l2"])
                    max_hidden_abs = max(max_hidden_abs, hidden["max_abs"])
                if logits["relative_l2"] is not None:
                    max_logits_relative = max(max_logits_relative, logits["relative_l2"])
                    max_logits_abs = max(max_logits_abs, logits["max_abs"])
                row_nonfinite = hidden["reference_nonfinite_count"] + hidden["candidate_nonfinite_count"] + logits["reference_nonfinite_count"] + logits["candidate_nonfinite_count"] > 0
                nonfinite_rows += int(row_nonfinite)
                greedy_exact = None if row_nonfinite else left["greedy_token_id"] == right["greedy_token_id"]
                greedy_mismatch += int(greedy_exact is False)
                left_top = {item["token_id"] for item in (left["topk"] or [])}
                right_top = {item["token_id"] for item in (right["topk"] or [])}
                overlap = len(left_top & right_top) if not row_nonfinite else None
                if overlap is not None:
                    overlap_min = min(overlap_min, overlap)
                reference_finite = hidden["reference_nonfinite_count"] + logits["reference_nonfinite_count"] == 0
                candidate_finite = hidden["candidate_nonfinite_count"] + logits["candidate_nonfinite_count"] == 0
                output_row = {"case_id": key[0], "step": key[1], "greedy_exact": greedy_exact, "top_k_overlap": overlap, "hidden": hidden, "logits": logits, "reference_finite": reference_finite, "candidate_finite": candidate_finite}
                rows_out.write(json.dumps(output_row, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n")
                row_count += 1
        rows_sha = sha256_file(result_rows, "comparison rows")
        manifest = {"schema_version": SCHEMA, "status": "valid" if nonfinite_rows == 0 else "blocked", "promotion_eligible": False, "created_utc": _VALIDATOR.legacy_oracle.utc_now(), "compare_kind": compare_kind, "reference": {"path": str(reference["root"].resolve()), "manifest_sha256": reference["manifest_sha256"], "schema_version": reference["manifest"]["schema_version"], "oracle_kind": reference["manifest"]["oracle_kind"]}, "candidate": {"path": str(candidate["root"].resolve()), "manifest_sha256": candidate["manifest_sha256"], "schema_version": candidate["manifest"]["schema_version"], "oracle_kind": candidate["manifest"]["oracle_kind"]}, "vector_contract": {"hidden_shape": [HIDDEN_SIZE], "logits_shape": [VOCAB_SIZE], "dtype": "f32", "endianness": "little", "metric_denominator": "max(reference_l2,1e-30)", "top_k": TOP_K}, "rows": {"file": "rows.jsonl", "record_count": row_count, "sha256": rows_sha}, "summary": {"row_count": row_count, "nonfinite_rows": nonfinite_rows, "greedy_mismatch_rows": greedy_mismatch, "max_hidden_relative_l2": max_hidden_relative, "max_hidden_max_abs": max_hidden_abs, "max_logits_relative_l2": max_logits_relative, "max_logits_max_abs": max_logits_abs, "minimum_top_k_overlap": None if overlap_min == TOP_K and nonfinite_rows else overlap_min}, "observed_values_only": True}
        (temporary / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        sums = []
        for path in sorted(temporary.rglob("*")):
            if path.is_file() and path.name != "SHA256SUMS":
                sums.append(f"{sha256_file(path, path.name)}  {path.relative_to(temporary).as_posix()}\n")
        (temporary / "SHA256SUMS").write_text("".join(sums), encoding="ascii")
        verify_loaded(reference)
        verify_loaded(candidate)
        for path in (result_rows, temporary / "manifest.json", temporary / "SHA256SUMS"):
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        directory_fd = os.open(temporary, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        publish_noreplace(temporary, output)
        return manifest
    finally:
        if temporary.exists():
            import shutil
            shutil.rmtree(temporary, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--compare-kind", choices=("source_gate", "path_gate"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        reference = load_artifact(args.reference)
        candidate = load_artifact(args.candidate)
        result = compare(reference, candidate, args.compare_kind, args.output)
        print(json.dumps(result, ensure_ascii=True, sort_keys=True))
        return 0
    except (ComparisonError, OSError, ValueError) as error:
        print(f"Qwen3.5 calibration comparison failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
