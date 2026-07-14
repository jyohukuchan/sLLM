#!/usr/bin/env python3
"""Compare full-vector Qwen3.5 calibration sidecars without retaining rows.

``source_gate`` compares an independent BF16 source artifact with an AQ4
target sidecar.  ``path_gate`` compares AQ4 all-M=1 with an optimized AQ4
sidecar.  The two identities are intentionally checked separately and this
tool never creates or modifies a threshold policy from observed values.
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
    if path.is_symlink():
        raise ComparisonError(f"{label} must not be a symlink")
    try:
        info = path.lstat()
    except OSError as error:
        raise ComparisonError(f"{label} is unavailable: {error}") from error
    if not stat.S_ISREG(info.st_mode):
        raise ComparisonError(f"{label} must be a regular file")
    if max_bytes is not None and info.st_size > max_bytes:
        raise ComparisonError(f"{label} exceeds {max_bytes} bytes")
    return path


def stat_identity(path: Path) -> tuple[int, int, int, int, int]:
    info = path.lstat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_mode


def read_json(path: Path, label: str, max_bytes: int = 16 * 1024 * 1024) -> Any:
    regular(path, label, max_bytes)
    before = stat_identity(path)
    raw = path.read_bytes()
    after = stat_identity(path)
    if before != after:
        raise ComparisonError(f"{label} changed while being read")
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicate_keys, parse_constant=reject_nonfinite)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ComparisonError(f"invalid {label}: {error}") from error


def sha256_file(path: Path, label: str) -> str:
    regular(path, label)
    before = stat_identity(path)
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    if before != stat_identity(path):
        raise ComparisonError(f"{label} changed while being hashed")
    return digest.hexdigest()


def ensure_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ComparisonError(f"{label} must be a lowercase SHA-256 digest")
    return value


def relative_path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ComparisonError(f"{label} must be relative")
    return _VALIDATOR.relative_path(root, value, label)


def read_chunks(handle: Any, offset: int, elements: int, chunk_elements: int) -> Iterator[list[float]]:
    handle.seek(offset)
    remaining = elements
    while remaining:
        count = min(remaining, chunk_elements)
        raw = handle.read(count * F32_BYTES)
        if len(raw) != count * F32_BYTES:
            raise ComparisonError("vector sidecar ended before row boundary")
        yield list(struct.unpack(f"<{count}f", raw))
        remaining -= count


def finite_metrics(reference: Iterator[list[float]], candidate: Iterator[list[float]], elements: int, chunk_elements: int) -> dict[str, Any]:
    ref_sq = 0.0
    delta_sq = 0.0
    max_abs = 0.0
    ref_nonfinite = 0
    candidate_nonfinite = 0
    seen = 0
    while seen < elements:
        try:
            ref_chunk = next(reference)
            candidate_chunk = next(candidate)
        except StopIteration as error:
            raise ComparisonError("vector stream ended early") from error
        if len(ref_chunk) != len(candidate_chunk):
            raise ComparisonError("reference/candidate chunk lengths differ")
        for left, right in zip(ref_chunk, candidate_chunk):
            if not math.isfinite(left):
                ref_nonfinite += 1
            if not math.isfinite(right):
                candidate_nonfinite += 1
            if math.isfinite(left) and math.isfinite(right):
                difference = float(right) - float(left)
                max_abs = max(max_abs, abs(difference))
                delta_sq += difference * difference
                ref_sq += float(left) * float(left)
        seen += len(ref_chunk)
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
    expected = {"case_id", "step", "semantic_input_id", "observation", "input_token_ids_sha256", "hidden", "logits", "greedy_token_id", "topk", "finite"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ComparisonError(f"{label} fields differ")
    if not isinstance(value["case_id"], str) or not value["case_id"] or not isinstance(value["step"], int) or value["step"] < 0:
        raise ComparisonError(f"{label} identity is invalid")
    ensure_sha(value["input_token_ids_sha256"], f"{label}.input_token_ids_sha256")
    for name, elements in (("hidden", HIDDEN_SIZE), ("logits", VOCAB_SIZE)):
        item = value[name]
        if not isinstance(item, dict) or item.get("elements") != elements or item.get("bytes") != elements * F32_BYTES or item.get("dtype") != "f32" or item.get("endianness") != "little":
            raise ComparisonError(f"{label}.{name} shape contract differs")
        if not isinstance(item.get("offset_bytes"), int) or item["offset_bytes"] < 0 or item["offset_bytes"] % F32_BYTES:
            raise ComparisonError(f"{label}.{name}.offset_bytes is invalid")
        ensure_sha(item.get("sha256"), f"{label}.{name}.sha256")
    if not isinstance(value["greedy_token_id"], int) or not 0 <= value["greedy_token_id"] < VOCAB_SIZE or not isinstance(value["topk"], list) or len(value["topk"]) != TOP_K or not isinstance(value["finite"], bool):
        raise ComparisonError(f"{label} top-k/finite fields are invalid")
    return value


def load_artifact(root: Path) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise ComparisonError("artifact root must be a real directory")
    manifest = read_json(root / "manifest.json", "artifact manifest")
    if not isinstance(manifest, dict) or manifest.get("schema_version") not in {SOURCE_SCHEMA, TARGET_SCHEMA}:
        raise ComparisonError("artifact schema is unsupported")
    if manifest.get("oracle_kind") not in {"independent_source_full", "aq4_target", "same_artifact_all_m1", "aq4_optimized"}:
        raise ComparisonError("artifact oracle_kind is unsupported")
    identity = manifest.get("identity")
    if not isinstance(identity, dict) or identity.get("model_id") != "Qwen/Qwen3.5-9B":
        raise ComparisonError("artifact model identity differs")
    tokenizer = identity.get("tokenizer", {})
    ensure_sha(tokenizer.get("aggregate_sha256"), "artifact tokenizer aggregate")
    contract = manifest.get("vector_contract", {})
    if contract.get("hidden_shape") != [HIDDEN_SIZE] or contract.get("logits_shape") != [VOCAB_SIZE] or contract.get("dtype") != "f32" or contract.get("endianness") != "little":
        raise ComparisonError("artifact vector contract differs")
    chunk_elements = contract.get("chunk_elements")
    if not isinstance(chunk_elements, int) or not 0 < chunk_elements <= MAX_CHUNK_ELEMENTS:
        raise ComparisonError("artifact chunk size is invalid")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != {"rows", "hidden", "logits"}:
        raise ComparisonError("artifact files differ")
    rows_path = relative_path(root, files["rows"], "artifact rows")
    hidden_path = relative_path(root, files["hidden"], "artifact hidden")
    logits_path = relative_path(root, files["logits"], "artifact logits")
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    with rows_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                row = parse_row(json.loads(line, object_pairs_hook=reject_duplicate_keys, parse_constant=reject_nonfinite), f"rows[{line_number}]")
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ComparisonError(f"invalid rows JSON at line {line_number}: {error}") from error
            key = (row["case_id"], row["step"])
            if key in rows:
                raise ComparisonError("artifact rows contain duplicate case/step")
            rows[key] = row
            if len(rows) > MAX_ROWS:
                raise ComparisonError("artifact rows exceed bound")
    if not rows:
        raise ComparisonError("artifact rows are empty")
    if hidden_path.stat().st_size != len(rows) * HIDDEN_SIZE * F32_BYTES or logits_path.stat().st_size != len(rows) * VOCAB_SIZE * F32_BYTES:
        raise ComparisonError("artifact sidecar size differs from row count")
    previous_hidden = previous_logits = 0
    chunk_elements = contract["chunk_elements"]
    with hidden_path.open("rb") as hidden, logits_path.open("rb") as logit:
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
            if row["logits"]["nonfinite_count"] == 0:
                ranked = topk_from_chunks(read_chunks(logit, row["logits"]["offset_bytes"], VOCAB_SIZE, chunk_elements), VOCAB_SIZE, TOP_K)
                if ranked != row["topk"]:
                    raise ComparisonError(f"artifact top-k differs for {key}")
    sums_path = regular(root / "SHA256SUMS", "artifact SHA256SUMS", max_bytes=8 * 1024 * 1024)
    sums: dict[str, str] = {}
    for line_number, line in enumerate(sums_path.read_text(encoding="ascii").splitlines(), 1):
        fields = line.split("  ", 1)
        if len(fields) != 2 or fields[1] in sums:
            raise ComparisonError(f"artifact SHA256SUMS line {line_number} is invalid")
        sums[fields[1]] = ensure_sha(fields[0], f"artifact SHA256SUMS line {line_number}")
    expected_files = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() and not path.is_symlink() and path.name != "SHA256SUMS"}
    if set(sums) != expected_files:
        raise ComparisonError("artifact SHA256SUMS file set differs")
    for name, expected in sums.items():
        path = root / name
        if sha256_file(path, name) != expected:
            raise ComparisonError(f"artifact SHA256SUMS digest differs for {name}")
    return {"root": root, "manifest": manifest, "manifest_sha256": sha256_file(root / "manifest.json", "artifact manifest"), "rows": rows, "hidden": hidden_path, "logits": logits_path, "chunk_elements": chunk_elements}


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
        with reference["hidden"].open("rb") as reference_hidden, candidate["hidden"].open("rb") as candidate_hidden, reference["logits"].open("rb") as reference_logits, candidate["logits"].open("rb") as candidate_logits, result_rows.open("w", encoding="utf-8") as rows_out:
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
                greedy_exact = left["greedy_token_id"] == right["greedy_token_id"]
                greedy_mismatch += int(not greedy_exact)
                left_top = {item["token_id"] for item in left["topk"]}
                right_top = {item["token_id"] for item in right["topk"]}
                overlap = len(left_top & right_top) if not row_nonfinite else None
                if overlap is not None:
                    overlap_min = min(overlap_min, overlap)
                output_row = {"case_id": key[0], "step": key[1], "greedy_exact": greedy_exact, "top_k_overlap": overlap, "hidden": hidden, "logits": logits, "reference_finite": not row_nonfinite, "candidate_finite": not row_nonfinite}
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
        if os.path.lexists(output):
            raise ComparisonError(f"refusing to overwrite existing output: {output}")
        os.rename(temporary, output)
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
