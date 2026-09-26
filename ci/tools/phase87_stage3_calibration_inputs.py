#!/usr/bin/env python3
"""Build and validate the fixed Phase 87 Stage 3 calibration inputs.

The calibration corpus is deliberately kept outside ``mtp-bench-v1``.  This
tool makes that separation checkable: it validates the frozen MTP prompt and
sequence hashes, compares normalized text and all recorded sequence hashes,
and emits a deterministic JSONL capture input plus a digest manifest.

No model, tokenizer, GPU, or network access is required for the normal
``--check`` path.  A Qwen tokenizer may be supplied later to extend the hash
check to the rendered token ID stream before a GPU capture is started.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata
from typing import Any, Iterable


SCHEMA_VERSION = "phase87-stage3-calibration-manifest-v1"
SPEC_SCHEMA_VERSION = "phase87-stage3-calibration-spec-v1"
REPO = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE = REPO / "ci/fixtures/phase87-stage3-calibration-v1"
DEFAULT_MTP_MATRIX = REPO / "ci/matrix/mtp-bench-v1.json"
DEFAULT_MTP_FIXTURE = REPO / "ci/fixtures/mtp-bench-v1"
RENDERED_RELATIVE = Path("rendered/inputs.jsonl")
MANIFEST_RELATIVE = Path("manifest.json")
SUITE_RELATIVE = Path("suite.json")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# This is the reviewed Qwen3.8 single-user frame used by existing engine
# render tests.  The message is normalized before insertion so a prompt file's
# final newline cannot change the capture input.
TEMPLATE_PREFIX = "<|im_start|>user\n"
TEMPLATE_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
NORMALIZATION_DESCRIPTION = (
    "Unicode NFKC; CRLF/CR to LF; trailing ASCII spaces/tabs removed per line; "
    "terminal blank lines removed"
)


class CalibrationInputError(ValueError):
    """The fixed calibration corpus or an exclusion source is invalid."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def normalize_text(text: str) -> str:
    """Return the exact text form used by all overlap checks."""

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip(" \t") for line in text.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def token_ids_sha256(token_ids: Iterable[int]) -> str:
    """Match mtp-bench-v1's committed_token_sha256 derivation."""

    values = list(token_ids)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise CalibrationInputError("sequence committed_token_ids must be integers")
    return sha256_bytes(",".join(str(value) for value in values).encode("ascii"))


def _relative_safe(value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise CalibrationInputError(f"{name} must be a relative path")
    path = Path(value)
    if ".." in path.parts:
        raise CalibrationInputError(f"{name} may not escape its root: {value}")
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CalibrationInputError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise CalibrationInputError(f"{label} must be a JSON object")
    return value


def _read_text(path: Path, label: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise CalibrationInputError(f"cannot read {label}: {error}") from error


def _digest_file(path: Path, label: str) -> str:
    try:
        return sha256_bytes(path.read_bytes())
    except OSError as error:
        raise CalibrationInputError(f"cannot hash {label}: {error}") from error


def _source_tree(root: Path) -> tuple[list[dict[str, Any]], str]:
    if not root.is_dir():
        raise CalibrationInputError(f"fixture root is missing: {root}")
    entries: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        payload = path.read_bytes()
        entries.append({"path": relative, "size_bytes": len(payload), "sha256": sha256_bytes(payload)})
    return entries, sha256_bytes(canonical_json(entries))


def _walk_sequence_expectations(node: Any, output: dict[str, dict[str, Any]]) -> None:
    if isinstance(node, dict):
        if "sequence_file" in node:
            sequence_file = _relative_safe(node["sequence_file"], "sequence_file")
            if sequence_file in output:
                raise CalibrationInputError(f"duplicate MTP sequence reference: {sequence_file}")
            output[str(sequence_file)] = {
                "committed_token_sha256": node.get("committed_token_sha256"),
                "sequence_sha256": node.get("sequence_sha256"),
            }
        for value in node.values():
            _walk_sequence_expectations(value, output)
    elif isinstance(node, list):
        for value in node:
            _walk_sequence_expectations(value, output)


def _load_mtp_exclusions(matrix_path: Path, fixture_root: Path) -> dict[str, Any]:
    matrix = _read_json(matrix_path, "mtp-bench-v1 matrix")
    if matrix.get("schema_version") != "mtp-bench-v1":
        raise CalibrationInputError("unexpected mtp-bench-v1 matrix schema")
    conditions = matrix.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise CalibrationInputError("mtp-bench-v1 matrix has no conditions")

    prompt_records: list[dict[str, Any]] = []
    prompt_paths: set[str] = set()
    for condition in conditions:
        if not isinstance(condition, dict):
            raise CalibrationInputError("mtp-bench-v1 condition is not an object")
        prompt_rel = _relative_safe(condition.get("prompt_file"), "condition.prompt_file")
        prompt_key = prompt_rel.as_posix()
        prompt_path = fixture_root / prompt_rel
        if prompt_key in prompt_paths:
            raise CalibrationInputError(f"duplicate MTP prompt reference: {prompt_key}")
        prompt_paths.add(prompt_key)
        if not prompt_path.is_file():
            raise CalibrationInputError(f"MTP prompt is missing: {prompt_path}")
        raw = prompt_path.read_bytes()
        actual_raw = sha256_bytes(raw)
        expected_raw = condition.get("prompt_sha256")
        if expected_raw != actual_raw:
            raise CalibrationInputError(
                f"MTP prompt hash mismatch for {prompt_key}: expected {expected_raw}, got {actual_raw}"
            )
        text = raw.decode("utf-8")
        prompt_records.append(
            {
                "cond_id": condition.get("cond_id"),
                "path": prompt_key,
                "raw_sha256": actual_raw,
                "normalized_sha256": sha256_bytes(normalize_text(text).encode("utf-8")),
                "normalized_text": normalize_text(text),
            }
        )

    sequence_expectations: dict[str, dict[str, Any]] = {}
    _walk_sequence_expectations(matrix, sequence_expectations)
    if not sequence_expectations:
        raise CalibrationInputError("mtp-bench-v1 matrix has no sequence references")
    sequence_records: list[dict[str, Any]] = []
    committed_text_hashes: set[str] = set()
    committed_token_hashes: set[str] = set()
    for sequence_key in sorted(sequence_expectations):
        sequence_path = fixture_root / Path(sequence_key)
        if not sequence_path.is_file():
            raise CalibrationInputError(f"MTP sequence is missing: {sequence_path}")
        actual_sequence_hash = _digest_file(sequence_path, f"MTP sequence {sequence_key}")
        expected_sequence_hash = sequence_expectations[sequence_key].get("sequence_sha256")
        if expected_sequence_hash != actual_sequence_hash:
            raise CalibrationInputError(
                f"MTP sequence hash mismatch for {sequence_key}: expected {expected_sequence_hash}, got {actual_sequence_hash}"
            )
        sequence = _read_json(sequence_path, f"MTP sequence {sequence_key}")
        token_ids = sequence.get("committed_token_ids")
        if not isinstance(token_ids, list) or not token_ids:
            raise CalibrationInputError(f"MTP sequence has no committed tokens: {sequence_key}")
        actual_token_hash = token_ids_sha256(token_ids)
        expected = sequence_expectations[sequence_key].get("committed_token_sha256")
        if expected != actual_token_hash:
            raise CalibrationInputError(
                f"MTP committed token hash mismatch for {sequence_key}: expected {expected}, got {actual_token_hash}"
            )
        sequence_token_hash = sequence.get("committed_token_sha256", actual_token_hash)
        if not isinstance(sequence_token_hash, str) or not SHA256_RE.fullmatch(sequence_token_hash):
            raise CalibrationInputError(f"invalid committed token hash in {sequence_key}")
        output_text_hash = sequence.get("output_text_sha256")
        prompt_hash = sequence.get("prompt_sha256")
        for value, label in ((output_text_hash, "output_text_sha256"), (prompt_hash, "prompt_sha256")):
            if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
                raise CalibrationInputError(f"invalid {label} in {sequence_key}")
        committed_token_hashes.add(sequence_token_hash)
        committed_text_hashes.add(output_text_hash)
        committed_text_hashes.add(prompt_hash)
        sequence_records.append(
            {
                "path": sequence_key,
                "sequence_sha256": actual_sequence_hash,
                "committed_token_sha256": sequence_token_hash,
                "output_text_sha256": output_text_hash,
                "prompt_sha256": prompt_hash,
                "committed_token_count": len(token_ids),
            }
        )

    source_files, source_tree_sha256 = _source_tree(fixture_root)
    # Every UTF-8 source text in the evaluation fixture participates in the
    # normalized-text check. This includes corpus files, because a calibration
    # case must not accidentally become an exact copy of an evaluation source
    # that is later wrapped in a different instruction.
    source_texts: dict[str, list[str]] = {}
    source_raw_hashes: dict[str, list[str]] = {}
    for entry in source_files:
        path = fixture_root / entry["path"]
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        normalized = normalize_text(text)
        source_texts.setdefault(normalized, []).append(entry["path"])
        source_raw_hashes.setdefault(entry["sha256"], []).append(entry["path"])

    return {
        "matrix_sha256": _digest_file(matrix_path, "mtp-bench-v1 matrix"),
        "fixture_files": source_files,
        "fixture_tree_sha256": source_tree_sha256,
        "prompt_records": prompt_records,
        "sequence_records": sequence_records,
        "source_texts": source_texts,
        "source_raw_hashes": source_raw_hashes,
        "committed_text_hashes": sorted(committed_text_hashes),
        "committed_token_hashes": sorted(committed_token_hashes),
    }


def _check_case_disjointness(
    cases: list[dict[str, Any]], exclusions: dict[str, Any]
) -> list[dict[str, Any]]:
    source_texts: dict[str, list[str]] = exclusions["source_texts"]
    source_raw_hashes: dict[str, list[str]] = exclusions["source_raw_hashes"]
    committed_text_hashes = set(exclusions["committed_text_hashes"])
    committed_token_hashes = set(exclusions["committed_token_hashes"])
    seen_normalized: dict[str, str] = {}
    seen_hashes: dict[str, str] = {}
    records: list[dict[str, Any]] = []
    failures: list[str] = []

    for case in cases:
        case_id = case["id"]
        raw_sha = case["raw_prompt_sha256"]
        normalized_text = case["normalized_text"]
        normalized_sha = case["normalized_text_sha256"]
        rendered_text = case["rendered_prompt"]
        rendered_sha = case["rendered_prompt_sha256"]
        rendered_normalized_sha = sha256_bytes(normalize_text(rendered_text).encode("utf-8"))
        normalized_hits = list(source_texts.get(normalized_text, []))
        if normalized_hits:
            failures.append(f"{case_id}: normalized text overlaps {sorted(normalized_hits)}")
        if normalized_text in seen_normalized:
            failures.append(f"{case_id}: normalized text duplicates {seen_normalized[normalized_text]}")
        seen_normalized[normalized_text] = case_id

        candidate_hashes = {
            "raw_prompt_sha256": raw_sha,
            "normalized_text_sha256": normalized_sha,
            "rendered_prompt_sha256": rendered_sha,
            "rendered_normalized_sha256": rendered_normalized_sha,
        }
        evaluation_hash_hits: dict[str, list[str]] = {}
        committed_text_hits: dict[str, list[str]] = {}
        committed_token_hits: dict[str, list[str]] = {}
        for label, digest in candidate_hashes.items():
            if digest in source_raw_hashes:
                evaluation_hash_hits[label] = sorted(source_raw_hashes[digest])
            if digest in committed_text_hashes:
                committed_text_hits[label] = [digest]
            if digest in committed_token_hashes:
                committed_token_hits[label] = [digest]
            if digest in seen_hashes:
                failures.append(f"{case_id}: hash duplicates {seen_hashes[digest]}")
            seen_hashes[digest] = case_id
        if evaluation_hash_hits:
            failures.append(f"{case_id}: evaluation hash overlap {evaluation_hash_hits}")
        if committed_text_hits:
            failures.append(f"{case_id}: committed text hash overlap {committed_text_hits}")
        if committed_token_hits:
            failures.append(f"{case_id}: committed token hash overlap {committed_token_hits}")
        records.append(
            {
                "case_id": case_id,
                "normalized_text_overlap_paths": sorted(normalized_hits),
                "evaluation_hash_overlap": evaluation_hash_hits,
                "committed_text_hash_overlap": committed_text_hits,
                "committed_token_hash_overlap": committed_token_hits,
            }
        )

    if failures:
        raise CalibrationInputError("calibration corpus is not disjoint:\n- " + "\n- ".join(failures))
    return records


def _load_spec(
    fixture_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], str, str]:
    spec_path = fixture_root / "calibration-spec.json"
    spec = _read_json(spec_path, "calibration spec")
    if spec.get("schema_version") != SPEC_SCHEMA_VERSION or spec.get("state") != "frozen":
        raise CalibrationInputError("calibration spec is not a frozen v1 spec")
    cases = spec.get("cases")
    if not isinstance(cases, list) or not cases:
        raise CalibrationInputError("calibration spec has no cases")
    seen: set[str] = set()
    loaded: list[dict[str, Any]] = []
    for item in cases:
        if not isinstance(item, dict):
            raise CalibrationInputError("calibration case is not an object")
        case_id = item.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise CalibrationInputError(f"invalid or duplicate calibration case id: {case_id!r}")
        seen.add(case_id)
        prompt_rel = _relative_safe(item.get("prompt_file"), f"{case_id}.prompt_file")
        prompt_path = fixture_root / prompt_rel
        if not prompt_path.is_file():
            raise CalibrationInputError(f"calibration prompt is missing: {prompt_path}")
        raw_text = _read_text(prompt_path, f"calibration prompt {case_id}")
        normalized_text = normalize_text(raw_text)
        if not normalized_text:
            raise CalibrationInputError(f"calibration prompt is empty: {case_id}")
        max_new_tokens = item.get("max_new_tokens")
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise CalibrationInputError(f"invalid max_new_tokens for {case_id}")
        rendered = TEMPLATE_PREFIX + normalized_text + TEMPLATE_SUFFIX
        loaded.append(
            {
                "id": case_id,
                "family": item.get("family"),
                "language": item.get("language"),
                "prompt_file": prompt_rel.as_posix(),
                "max_new_tokens": max_new_tokens,
                "raw_prompt_sha256": sha256_bytes(raw_text.encode("utf-8")),
                "normalized_text": normalized_text,
                "normalized_text_sha256": sha256_bytes(normalized_text.encode("utf-8")),
                "rendered_prompt": rendered,
                "rendered_prompt_sha256": sha256_bytes(rendered.encode("utf-8")),
            }
        )
    suite_path = fixture_root / SUITE_RELATIVE
    suite = _read_json(suite_path, "Stage0 calibration suite")
    if set(suite) != {"seeds", "cases"}:
        raise CalibrationInputError("Stage0 calibration suite must contain only seeds and cases")
    seeds = suite.get("seeds")
    suite_cases = suite.get("cases")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
    ):
        raise CalibrationInputError("Stage0 calibration suite seeds must be nonempty u64 integers")
    if any(seed > (1 << 64) - 1 for seed in seeds):
        raise CalibrationInputError("Stage0 calibration suite seed is outside u64")
    if not isinstance(suite_cases, list) or len(suite_cases) != len(loaded):
        raise CalibrationInputError("Stage0 calibration suite case count does not match the spec")
    for loaded_case, suite_case in zip(loaded, suite_cases):
        if not isinstance(suite_case, dict) or set(suite_case) != {"id", "task", "language", "message"}:
            raise CalibrationInputError("Stage0 calibration suite case has the wrong shape")
        if suite_case["id"] != loaded_case["id"] or suite_case["language"] != loaded_case["language"]:
            raise CalibrationInputError(f"Stage0 suite identity mismatch for {loaded_case['id']}")
        if not isinstance(suite_case["task"], str) or not suite_case["task"]:
            raise CalibrationInputError(f"Stage0 suite task is empty for {loaded_case['id']}")
        if suite_case["message"] != loaded_case["normalized_text"]:
            raise CalibrationInputError(
                f"Stage0 suite message differs from prompt file for {loaded_case['id']}"
            )
        loaded_case["task"] = suite_case["task"]
    return (
        spec,
        suite,
        loaded,
        _digest_file(spec_path, "calibration spec"),
        _digest_file(suite_path, "Stage0 calibration suite"),
    )


def build_artifacts(
    repo: Path = REPO,
    fixture_root: Path = DEFAULT_FIXTURE,
    matrix_path: Path = DEFAULT_MTP_MATRIX,
    mtp_fixture_root: Path = DEFAULT_MTP_FIXTURE,
) -> tuple[bytes, bytes, dict[str, Any]]:
    """Return expected rendered JSONL bytes, manifest bytes, and a summary."""

    del repo  # Kept in the API so focused tests can pass an explicit workspace.
    spec, suite, loaded_cases, spec_sha256, suite_sha256 = _load_spec(fixture_root)
    exclusions = _load_mtp_exclusions(matrix_path, mtp_fixture_root)
    overlap_records = _check_case_disjointness(loaded_cases, exclusions)

    rendered_records: list[dict[str, Any]] = []
    rendered_lines: list[bytes] = []
    case_summaries: list[dict[str, Any]] = []
    for case, overlap in zip(loaded_cases, overlap_records):
        rendered_record = {
            "schema_version": "phase87-stage3-calibration-input-v1",
            "case_id": case["id"],
            "family": case["family"],
            "task": case["task"],
            "language": case["language"],
            "message": case["normalized_text"],
            "rendered_prompt": case["rendered_prompt"],
            "max_new_tokens": case["max_new_tokens"],
            "raw_prompt_sha256": case["raw_prompt_sha256"],
            "normalized_text_sha256": case["normalized_text_sha256"],
            "rendered_prompt_sha256": case["rendered_prompt_sha256"],
        }
        rendered_records.append(rendered_record)
        rendered_lines.append(canonical_json(rendered_record))
        case_summaries.append(
            {
                "case_id": case["id"],
                "family": case["family"],
                "task": case["task"],
                "language": case["language"],
                "prompt_file": case["prompt_file"],
                "max_new_tokens": case["max_new_tokens"],
                "raw_prompt_sha256": case["raw_prompt_sha256"],
                "normalized_text_sha256": case["normalized_text_sha256"],
                "rendered_prompt_sha256": case["rendered_prompt_sha256"],
                "rendered_chars": len(case["rendered_prompt"]),
                "overlap": overlap,
            }
        )
    rendered_bytes = b"".join(rendered_lines)

    source_text_hashes = sorted(
        {
            entry["normalized_sha256"]
            for entry in exclusions["prompt_records"]
        }
    )
    # Include every source file's raw digest in the manifest, while keeping the
    # prompt-specific normalized digest list explicit and easy to audit.
    manifest_without_digest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "state": "PASS",
        "fixture_id": "phase87-stage3-calibration-v1",
        "purpose": spec["purpose"],
        "model": spec["model"],
        "generator": "ci/tools/phase87_stage3_calibration_inputs.py",
        "normalization": NORMALIZATION_DESCRIPTION,
        "rendering": {
            "template_id": spec["capture_contract"]["chat_template"],
            "prefix": TEMPLATE_PREFIX,
            "suffix": TEMPLATE_SUFFIX,
            "message_uses_normalized_text": True,
        },
        "capture_contract": spec["capture_contract"],
        "seeds": suite["seeds"],
        "source": {
            "spec": {"path": "calibration-spec.json", "sha256": spec_sha256},
            "stage0_suite": {"path": SUITE_RELATIVE.as_posix(), "sha256": suite_sha256},
            "mtp_matrix": {"path": "ci/matrix/mtp-bench-v1.json", "sha256": exclusions["matrix_sha256"]},
            "mtp_fixture_root": {
                "path": "ci/fixtures/mtp-bench-v1",
                "file_count": len(exclusions["fixture_files"]),
                "tree_sha256": exclusions["fixture_tree_sha256"],
                "files": exclusions["fixture_files"],
            },
        },
        "exclusion": {
            "exact_normalized_text": {
                "evaluation_prompt_count": len(exclusions["prompt_records"]),
                "evaluation_prompt_sha256": source_text_hashes,
                "scope": "all UTF-8 files under mtp-bench-v1, with prompt and corpus text included",
            },
            "hash_domains": {
                "evaluation_raw_file_sha256": sorted(exclusions["source_raw_hashes"]),
                "committed_output_text_sha256": sorted(
                    record["output_text_sha256"] for record in exclusions["sequence_records"]
                ),
                "committed_prompt_sha256": sorted(
                    record["prompt_sha256"] for record in exclusions["sequence_records"]
                ),
                "committed_token_sha256": exclusions["committed_token_hashes"],
            },
            "overlap_result": {
                "case_count": len(overlap_records),
                "normalized_text_overlap_count": 0,
                "evaluation_hash_overlap_count": 0,
                "committed_text_hash_overlap_count": 0,
                "committed_token_hash_overlap_count": 0,
            },
        },
        "cases": case_summaries,
        "rendered_inputs": {
            "path": RENDERED_RELATIVE.as_posix(),
            "format": "UTF-8 JSONL; one canonical JSON object per case in spec order",
            "case_count": len(rendered_records),
            "sha256": sha256_bytes(rendered_bytes),
        },
    }
    manifest_digest = sha256_bytes(canonical_json(manifest_without_digest))
    manifest = dict(manifest_without_digest)
    manifest["manifest_payload_sha256"] = manifest_digest
    manifest_bytes = canonical_json(manifest)
    summary = {
        "manifest": manifest,
        "manifest_bytes": manifest_bytes,
        "rendered_bytes": rendered_bytes,
        "cases": loaded_cases,
        "sequence_records": exclusions["sequence_records"],
    }
    return rendered_bytes, manifest_bytes, summary


def _check_tokenizer_overlap(
    tokenizer_path: Path, cases: list[dict[str, Any]], sequence_records: list[dict[str, Any]]
) -> dict[str, Any]:
    try:
        from tokenizers import Tokenizer
    except ImportError as error:  # pragma: no cover - environment dependent
        raise CalibrationInputError("--tokenizer requires the tokenizers package") from error
    try:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    except Exception as error:  # noqa: BLE001 - convert library errors to contract errors
        raise CalibrationInputError(f"cannot load tokenizer: {error}") from error
    committed = {record["committed_token_sha256"] for record in sequence_records}
    collisions: list[dict[str, str]] = []
    case_hashes: list[dict[str, str]] = []
    for case in cases:
        token_ids = tokenizer.encode(case["rendered_prompt"], add_special_tokens=False).ids
        digest = token_ids_sha256(token_ids)
        case_hashes.append({"case_id": case["id"], "rendered_token_sha256": digest})
        if digest in committed:
            collisions.append({"case_id": case["id"], "rendered_token_sha256": digest})
    if collisions:
        raise CalibrationInputError(f"rendered token hash overlaps committed MTP tokens: {collisions}")
    return {
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_sha256": _digest_file(tokenizer_path, "tokenizer"),
        "case_hashes": case_hashes,
        "committed_token_overlap_count": 0,
    }


def _write_or_check(
    fixture_root: Path, rendered_bytes: bytes, manifest_bytes: bytes, *, write: bool
) -> None:
    rendered_path = fixture_root / RENDERED_RELATIVE
    manifest_path = fixture_root / MANIFEST_RELATIVE
    if write:
        rendered_path.parent.mkdir(parents=True, exist_ok=True)
        rendered_path.write_bytes(rendered_bytes)
        manifest_path.write_bytes(manifest_bytes)
        return
    for path, expected, label in (
        (rendered_path, rendered_bytes, "rendered input"),
        (manifest_path, manifest_bytes, "calibration manifest"),
    ):
        if not path.is_file():
            raise CalibrationInputError(f"{label} is missing: {path}; run with --write")
        actual = path.read_bytes()
        if actual != expected:
            raise CalibrationInputError(
                f"{label} is stale or nondeterministic: {path} (expected sha256 {sha256_bytes(expected)}, got {sha256_bytes(actual)})"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--mtp-matrix", type=Path, default=DEFAULT_MTP_MATRIX)
    parser.add_argument("--mtp-fixture-dir", type=Path, default=DEFAULT_MTP_FIXTURE)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="write deterministic manifest and rendered JSONL")
    mode.add_argument("--check", action="store_true", help="validate checked-in manifest and rendered JSONL")
    parser.add_argument("--tokenizer", type=Path, help="optional Qwen tokenizer.json for rendered token hash checks")
    args = parser.parse_args(argv)
    try:
        rendered_bytes, manifest_bytes, summary = build_artifacts(
            fixture_root=args.fixture_dir.resolve(),
            matrix_path=args.mtp_matrix.resolve(),
            mtp_fixture_root=args.mtp_fixture_dir.resolve(),
        )
        tokenizer_summary = None
        if args.tokenizer is not None:
            tokenizer_summary = _check_tokenizer_overlap(
                args.tokenizer.resolve(), summary["cases"], summary["sequence_records"]
            )
        _write_or_check(args.fixture_dir.resolve(), rendered_bytes, manifest_bytes, write=args.write)
    except CalibrationInputError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "state": "PASS",
                "fixture": str(args.fixture_dir),
                "cases": len(summary["cases"]),
                "rendered_inputs_sha256": sha256_bytes(rendered_bytes),
                "manifest_sha256": sha256_bytes(manifest_bytes),
                "manifest_payload_sha256": summary["manifest"]["manifest_payload_sha256"],
                "tokenizer_check": tokenizer_summary,
                "write": args.write,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
