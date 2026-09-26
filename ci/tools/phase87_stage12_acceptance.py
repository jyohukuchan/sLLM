#!/usr/bin/env python3
"""Validate the Stage 12 MTP identity and derive width 2/3/4 M1 values.

The v2 benchmark deliberately points at the existing v1 Tier A fixture tree.
This module validates that indirection before reading a GPU report.  It also
accepts the compact ``per_prompt`` form emitted by a host or GPU runner and,
for compatibility with the existing logits capture, can derive the same
values from raw draft/target rows through ``mtp_expected_acceptance.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import struct
from typing import Any


REPO = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO / "ci/matrix/mtp-bench-v2.json"
WIDTHS = (2, 3, 4)
UUIDS = {"gfx1030": "GPU-76a08c022586fed6", "gfx1201": "GPU-a8e9ddefa2d60f55"}
QWEN38_NVFP4_COMPANION_ENCODING = "nvfp4-w4a4-e2m1-block16-e4m3fn-f32"
QWEN38_NVFP4_COMPANION_DIGEST = "sha256:d9698c41954ef7b53a2937c0f662ac2a273f1bdc40c602f77d4928b63de991e1"
QWEN38_DRAFT_VOCAB_SHA256 = "24bff6b41785a7729bff183dfea7997e6446173e0df7254cc5761a7519fdebd0"


class Stage12IdentityError(ValueError):
    """The report or reused v1 input is not the frozen Stage 12 identity."""


def sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise Stage12IdentityError(f"cannot read {path}: {error}") from error


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _fixture_tree_sha256(root: Path) -> str:
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        files.append({
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    return hashlib.sha256(_canonical_json(files)).hexdigest()


def _relative_path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise Stage12IdentityError(f"{label} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise Stage12IdentityError(f"{label} escapes its root: {value}")
    return root / path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Stage12IdentityError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise Stage12IdentityError(f"{label} must be a JSON object")
    return value


def validate_width(width: int) -> int:
    if isinstance(width, bool) or not isinstance(width, int) or width not in WIDTHS:
        raise Stage12IdentityError("MTP width must be exactly 2, 3 or 4")
    return width


def _valid_digest(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _le_i32_sha256(tokens: Any) -> str:
    if not isinstance(tokens, list) or not tokens or any(
        isinstance(token, bool) or not isinstance(token, int) or token < 0 or token >= 248_320
        for token in tokens
    ):
        raise Stage12IdentityError("frozen prefix tokens are not a valid nonempty i32 list")
    return hashlib.sha256(b"".join(struct.pack("<i", token) for token in tokens)).hexdigest()


def load_identity(manifest_path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Load v2 and validate its references to the immutable v1 benchmark."""

    manifest_path = manifest_path.resolve()
    manifest = _read_json(manifest_path, "mtp-bench-v2 manifest")
    if manifest.get("schema_version") != "mtp-bench-v2" or manifest.get("state") != "frozen":
        raise Stage12IdentityError("mtp-bench-v2 manifest is not frozen")
    source = manifest.get("source_manifest")
    if not isinstance(source, dict):
        raise Stage12IdentityError("v2 source_manifest is missing")
    source_path = _relative_path(REPO, source.get("path"), "source_manifest.path")
    source_digest = source.get("sha256")
    if source_digest != sha256(source_path):
        raise Stage12IdentityError("v1 source manifest SHA-256 does not match v2")
    v1 = _read_json(source_path, "mtp-bench-v1 manifest")
    if v1.get("schema_version") != "mtp-bench-v1" or v1.get("state") != "frozen":
        raise Stage12IdentityError("v2 source is not the frozen mtp-bench-v1 manifest")
    fixture_root = _relative_path(REPO, manifest.get("fixture_root"), "fixture_root")
    if not fixture_root.is_dir():
        raise Stage12IdentityError(f"v1 fixture root is missing: {fixture_root}")
    expected_tree = manifest.get("fixture_tree_sha256")
    if expected_tree != _fixture_tree_sha256(fixture_root):
        raise Stage12IdentityError("v1 fixture tree SHA-256 does not match v2")
    conditions = [condition for condition in v1.get("conditions", []) if condition.get("tier") == "A"]
    if len(conditions) != manifest.get("tier_a_count") or len(conditions) != 26:
        raise Stage12IdentityError("v2 requires exactly the 26 frozen Tier A conditions")
    by_case: dict[str, dict[str, Any]] = {}
    for condition in conditions:
        case_id = condition.get("cond_id")
        if not isinstance(case_id, str) or case_id in by_case:
            raise Stage12IdentityError(f"invalid or duplicate Tier A condition: {case_id!r}")
        prompt = _relative_path(fixture_root, condition.get("prompt_file"), f"{case_id}.prompt_file")
        if not prompt.is_file() or condition.get("prompt_sha256") != sha256(prompt):
            raise Stage12IdentityError(f"Tier A prompt hash mismatch: {case_id}")
        by_case[case_id] = condition
    frozen = manifest.get("frozen_prefix_identity")
    if not isinstance(frozen, dict) or not isinstance(frozen.get("entries"), dict):
        raise Stage12IdentityError("v2 frozen prefix identity is missing")
    expected_prefixes = frozen["entries"]
    if set(expected_prefixes) != set(by_case):
        raise Stage12IdentityError("v2 frozen prefix case set differs from Tier A")
    for case_id, condition in by_case.items():
        item = expected_prefixes[case_id]
        if not isinstance(item, dict):
            raise Stage12IdentityError(f"invalid frozen prefix identity: {case_id}")
        prompt_count = item.get("prompt_token_count")
        output_count = item.get("output_prefix_token_count")
        if (isinstance(prompt_count, bool) or not isinstance(prompt_count, int)
                or prompt_count != condition["prompt_tokens_rendered"]
                or isinstance(output_count, bool) or not isinstance(output_count, int)
                or not 0 < output_count <= condition["output_tokens"]
                or not all(_valid_digest(item.get(key)) for key in (
                    "prompt_tokens_le_i32_sha256", "output_prefix_tokens_le_i32_sha256",
                    "output_prefix_tokens_decimal_csv_sha256"
                ))):
            raise Stage12IdentityError(f"frozen prefix shape or hash differs: {case_id}")
        series = item.get("reference_series")
        if series in condition["reference_sequences"]:
            if item["output_prefix_tokens_decimal_csv_sha256"] != condition["reference_sequences"][series]["committed_token_sha256"]:
                raise Stage12IdentityError(f"frozen reference sequence differs: {case_id}")
        elif not (case_id == "code-python-review-zh" and series == "phase86-frozen-existing-review"):
            raise Stage12IdentityError(f"unknown frozen prefix reference exception: {case_id}")
    prefix_path = _relative_path(REPO, frozen.get("source_path"), "frozen_prefix_identity.source_path")
    if prefix_path.is_file():
        if sha256(prefix_path) != frozen.get("source_sha256"):
            raise Stage12IdentityError("frozen Phase86 prefix file hash changed")
        prefix_document = _read_json(prefix_path, "frozen Phase86 prefix file")
        rows = prefix_document.get("entries")
        if not isinstance(rows, list) or len(rows) != 26:
            raise Stage12IdentityError("frozen Phase86 prefix row count differs")
        if {row.get("case_id") for row in rows if isinstance(row, dict)} != set(expected_prefixes):
            raise Stage12IdentityError("frozen Phase86 prefix case set differs")
        for row in rows:
            case_id = row.get("case_id")
            if case_id not in expected_prefixes:
                raise Stage12IdentityError(f"unknown frozen Phase86 prefix: {case_id}")
            expected = expected_prefixes[case_id]
            if (_le_i32_sha256(row.get("prompt_tokens")) != expected["prompt_tokens_le_i32_sha256"]
                    or _le_i32_sha256(row.get("output_prefix_tokens")) != expected["output_prefix_tokens_le_i32_sha256"]):
                raise Stage12IdentityError(f"frozen Phase86 prefix token hash changed: {case_id}")
    runtime = manifest.get("runtime_contract", {})
    if runtime.get("mtp_widths") != list(WIDTHS) or runtime.get("exact_targets") != ["gfx1030", "gfx1201"]:
        raise Stage12IdentityError("v2 runtime width or target contract differs")
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256(manifest_path),
        "source_manifest_sha256": source_digest,
        "source_manifest_path": source_path,
        "fixture_root": fixture_root,
        "conditions": by_case,
        "frozen_prefixes": expected_prefixes,
    }


def expected_tokens_per_block(steps: dict[str, float] | list[float], width: int) -> float:
    """Return ``1 + a1 + a1*a2 + ...`` for the selected proposal width."""

    validate_width(width)
    if isinstance(steps, dict):
        values = []
        for index in range(1, width + 1):
            key = f"step{index}"
            if key not in steps:
                raise Stage12IdentityError(f"missing proposal step: {key}")
            values.append(steps[key])
    else:
        values = list(steps)
        if len(values) != width:
            raise Stage12IdentityError(f"expected {width} proposal steps, got {len(values)}")
    total = 1.0
    product = 1.0
    for index, value in enumerate(values, 1):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
            raise Stage12IdentityError(f"proposal step {index} acceptance must be within [0,1]")
        product *= float(value)
        total += product
    return total


def _find_report_entries(node: Any) -> list[dict[str, Any]] | None:
    if isinstance(node, dict):
        entries = node.get("per_prompt")
        if isinstance(entries, list) and all(isinstance(item, dict) for item in entries):
            return entries
        entries = node.get("entries")
        if isinstance(entries, list) and any(
            isinstance(item, dict) and "draft_logit_rows" in item for item in entries
        ):
            return entries
        for value in node.values():
            found = _find_report_entries(value)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = _find_report_entries(value)
            if found is not None:
                return found
    return None


def _report_width(document: dict[str, Any]) -> int | None:
    values = []
    for node in (document, document.get("phase86", {}), document.get("mtp", {})):
        if isinstance(node, dict):
            for key in ("width", "mtp_width", "draft_width"):
                if key in node:
                    values.append(node[key])
    if not values:
        return None
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise Stage12IdentityError("M1 report width must be an integer")
    if len(set(values)) != 1:
        raise Stage12IdentityError("M1 report contains mismatched width identities")
    return values[0]


def _validate_report_header(document: dict[str, Any], identity: dict[str, Any], width: int) -> None:
    state = document.get("state")
    if state is not None and state != "PASS":
        raise Stage12IdentityError("M1 report is not PASS")
    cleanup = document.get("cleanup")
    if isinstance(cleanup, dict) and cleanup.get("zero") is not True:
        raise Stage12IdentityError("M1 report cleanup is not zero")
    target = document.get("target")
    if target not in UUIDS:
        raise Stage12IdentityError("M1 report target is not an exact tested GPU")
    model = document.get("model_sha256")
    if not isinstance(model, str) or not _valid_digest(model.removeprefix("sha256:")):
        raise Stage12IdentityError("M1 model identity is missing or invalid")
    report_width = _report_width(document)
    if report_width is not None and report_width != width:
        raise Stage12IdentityError(f"M1 report width {report_width} differs from requested {width}")
    phase86 = document.get("phase86")
    if isinstance(phase86, dict) and (
        phase86.get("schema_version") != "phase87-stage12-mtp-catch-up-v1"
        or phase86.get("mode") != "P"
        or phase86.get("width") != width
        or phase86.get("state") != "PASS"
        or phase86.get("all_valid") is not True
    ):
        raise Stage12IdentityError("M1 raw fixed-column report is outside the Stage12 P-mode contract")
    if isinstance(phase86, dict) and (
        document.get("companion_encoding") != QWEN38_NVFP4_COMPANION_ENCODING
        or document.get("companion_digest") != QWEN38_NVFP4_COMPANION_DIGEST
        or document.get("draft_vocab_sha256") != QWEN38_DRAFT_VOCAB_SHA256
    ):
        raise Stage12IdentityError("M1 raw companion or reduced draft vocabulary identity differs")
    if isinstance(phase86, dict) and phase86.get("prefix_file_sha256") != (
        "sha256:" + identity["manifest"]["frozen_prefix_identity"]["source_sha256"]
    ):
        raise Stage12IdentityError("M1 raw prefix file differs from the frozen Phase86 input")
    for key in ("manifest_sha256", "benchmark_manifest_sha256"):
        if key in document and document[key] != identity["manifest_sha256"]:
            raise Stage12IdentityError(f"M1 report {key} differs from v2 manifest")
    transform = document.get("support_transform")
    if isinstance(transform, dict) and (
        transform.get("top_k") != 20 or transform.get("top_p") != 0.95
    ):
        raise Stage12IdentityError("M1 support transform differs from the frozen selector")


def _normalise_prompt_steps(
    rows: list[dict[str, Any]], width: int, side: str | None
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for row in rows:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or case_id in output:
            raise Stage12IdentityError(f"invalid or duplicate M1 prompt: {case_id!r}")
        source = row.get("by_proposal_step")
        if not isinstance(source, dict):
            raise Stage12IdentityError(f"M1 prompt {case_id} has no by_proposal_step")
        values: dict[str, float] = {}
        for index in range(1, width + 1):
            key = f"step{index}"
            if key not in source:
                raise Stage12IdentityError(f"{case_id}: missing proposal step {key}")
            value: Any = source[key]
            if isinstance(value, dict):
                if side is None:
                    raise Stage12IdentityError(f"{case_id}: side is required for paired M1 values")
                value = value.get(side)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                raise Stage12IdentityError(f"{case_id}: {key} acceptance is outside [0,1]")
            values[key] = float(value)
        # Extra rows are a width identity error. It catches width-2 data fed to
        # a width-3/4 calculation even when the requested prefix is complete.
        extra = set(source) - set(values)
        if extra:
            raise Stage12IdentityError(f"{case_id}: unexpected proposal steps {sorted(extra)}")
        output[case_id] = values
    return output


def _load_compact_m1(
    document: dict[str, Any], identity: dict[str, Any], width: int, side: str | None
) -> dict[str, dict[str, float]] | None:
    rows = document.get("per_prompt")
    if not isinstance(rows, list):
        return None
    return _normalise_prompt_steps(rows, width, side)


def _load_logits_helper():
    path = REPO / "ci/tools/mtp_expected_acceptance.py"
    spec = importlib.util.spec_from_file_location("sllm_mtp_expected_acceptance", path)
    if spec is None or spec.loader is None:
        raise Stage12IdentityError("cannot load existing M1 acceptance helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_raw_m1(
    document: dict[str, Any], width: int, map_path: Path | None
) -> dict[str, dict[str, float]]:
    helper = _load_logits_helper()
    rows = _find_report_entries(document)
    if rows is None:
        raise Stage12IdentityError("M1 report has neither per_prompt nor logits entries")
    token_map = helper.load_candidate_draft_vocab_map(map_path.resolve()) if map_path else None
    values: dict[str, dict[str, list[float]]] = {}
    for entry in rows:
        case_id = entry.get("case_id")
        if not isinstance(case_id, str) or case_id in values:
            raise Stage12IdentityError(f"invalid or duplicate raw M1 prompt: {case_id!r}")
        try:
            scored = helper.run_rows(entry, 20, 0.95, token_map)
        except (KeyError, ValueError, OSError) as error:
            raise Stage12IdentityError(f"cannot score raw M1 prompt {case_id}: {error}") from error
        grouped: dict[str, list[float]] = {f"step{index}": [] for index in range(1, width + 1)}
        for step, value in scored:
            key = f"step{step + 1}"
            if key not in grouped:
                raise Stage12IdentityError(f"{case_id}: raw M1 has proposal step {step + 1} outside width {width}")
            grouped[key].append(float(value))
        if any(not grouped[key] for key in grouped):
            missing = [key for key, values_for_step in grouped.items() if not values_for_step]
            raise Stage12IdentityError(f"{case_id}: missing raw proposal steps {missing}")
        values[case_id] = {key: float(statistics.mean(step_values)) for key, step_values in grouped.items()}
    return values


def _report_prompt_steps(
    path: Path, identity: dict[str, Any], width: int, side: str | None, map_path: Path | None = None
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    document = _read_json(path, f"M1 report {path}")
    _validate_report_header(document, identity, width)
    if isinstance(document.get("phase86"), dict):
        execution_path = path.parent.parent / "execution.json"
        if not execution_path.is_file():
            raise Stage12IdentityError("M1 exact-GPU execution identity is missing")
        execution = _read_json(execution_path, "M1 execution identity")
        jobs = execution.get("jobs")
        matching = [job for job in jobs if job.get("name") == path.parent.name] if isinstance(jobs, list) else []
        if (execution.get("state") != "complete"
                or execution.get("uuid") != UUIDS[document["target"]]
                or execution.get("performance_level_restored") is not True
                or (document["target"] == "gfx1201" and execution.get("service_restored") is not True)
                or not _valid_digest(execution.get("binary_sha256"))
                or len(matching) != 1 or matching[0].get("exit_code") != 0
                or matching[0].get("report_sha256") != sha256(path)):
            raise Stage12IdentityError("M1 report differs from the completed exact-GPU job")
    compact = _load_compact_m1(document, identity, width, side)
    values = compact if compact is not None else _load_raw_m1(document, width, map_path)
    expected = set(identity["conditions"])
    if set(values) != expected:
        raise Stage12IdentityError(
            f"M1 report prompt set differs from the 26 Tier A conditions: {sorted(set(values) ^ expected)}"
        )
    if isinstance(document.get("phase86"), dict):
        for case_id in expected:
            row = _find_prompt_entry(document, case_id)
            frozen = identity["frozen_prefixes"][case_id]
            if row is None or (
                row.get("prompt_sha256") != "sha256:" + frozen["prompt_tokens_le_i32_sha256"]
                or row.get("output_prefix_sha256") != "sha256:" + frozen["output_prefix_tokens_le_i32_sha256"]
                or row.get("prompt_token_count") != frozen["prompt_token_count"]
                or row.get("output_prefix_token_count") != frozen["output_prefix_token_count"]
            ):
                raise Stage12IdentityError(f"M1 {case_id} differs from the frozen prompt/token column")
    else:
        for case_id in expected:
            row = _find_prompt_entry(document, case_id)
            frozen = identity["frozen_prefixes"][case_id]
            if row is None:
                raise Stage12IdentityError(f"M1 {case_id} compact prompt identity is missing")
            prompt_key = f"prompt_sha256_{side}" if side is not None else "prompt_sha256"
            output_key = f"output_prefix_sha256_{side}" if side is not None else "output_prefix_sha256"
            prompt_hash = row.get(prompt_key, row.get("prompt_sha256"))
            output_hash = row.get(output_key, row.get("output_prefix_sha256"))
            if (prompt_hash != "sha256:" + frozen["prompt_tokens_le_i32_sha256"]
                    or output_hash != "sha256:" + frozen["output_prefix_tokens_le_i32_sha256"]):
                raise Stage12IdentityError(f"M1 {case_id} compact prompt/token hash differs")
    return document, values


def calculate(
    baseline: Path,
    candidate: Path,
    width: int,
    manifest_path: Path = DEFAULT_MANIFEST,
    baseline_map: Path | None = None,
    candidate_map: Path | None = None,
) -> dict[str, Any]:
    """Calculate paired per-prompt expected acceptance for one width."""

    width = validate_width(width)
    identity = load_identity(manifest_path)
    left, baseline_values = _report_prompt_steps(baseline, identity, width, "baseline", baseline_map)
    right, candidate_values = _report_prompt_steps(candidate, identity, width, "candidate", candidate_map)
    if left.get("target") != right.get("target"):
        raise Stage12IdentityError("baseline and candidate M1 targets differ")
    if left.get("model_sha256") != right.get("model_sha256"):
        raise Stage12IdentityError("baseline and candidate M1 model identities differ")
    per_prompt = []
    for case_id in sorted(identity["conditions"]):
        b = baseline_values[case_id]
        c = candidate_values[case_id]
        frozen = identity["frozen_prefixes"][case_id]
        baseline_prompt = _find_prompt_entry(left, case_id)
        candidate_prompt = _find_prompt_entry(right, case_id)
        expected_tokens = identity["conditions"][case_id]["prompt_tokens_rendered"]
        if baseline_prompt is not None and candidate_prompt is not None:
            for key in ("prompt_sha256", "output_prefix_sha256", "prompt_token_count"):
                if baseline_prompt.get(key) != candidate_prompt.get(key):
                    raise Stage12IdentityError(f"M1 {case_id} baseline/candidate {key} differs")
            if (baseline_prompt.get("prompt_token_count") is not None
                    and baseline_prompt.get("prompt_token_count") != expected_tokens):
                raise Stage12IdentityError(f"M1 {case_id} rendered prompt token count differs from v1")
            if isinstance(left.get("phase86"), dict) and baseline_prompt.get("prompt_token_count") is None:
                raise Stage12IdentityError(f"M1 {case_id} raw prompt token count is missing")
        elif isinstance(left.get("phase86"), dict) or isinstance(right.get("phase86"), dict):
            raise Stage12IdentityError(f"M1 {case_id} raw prompt identity is missing")
        per_prompt.append({
            "case_id": case_id,
            "prompt_sha256_baseline": "sha256:" + frozen["prompt_tokens_le_i32_sha256"],
            "prompt_sha256_candidate": "sha256:" + frozen["prompt_tokens_le_i32_sha256"],
            "by_proposal_step": {
                f"step{index}": {"baseline": b[f"step{index}"], "candidate": c[f"step{index}"]}
                for index in range(1, width + 1)
            },
            "baseline_expected_tokens_per_block": expected_tokens_per_block(b, width),
            "candidate_expected_tokens_per_block": expected_tokens_per_block(c, width),
        })
    return {
        "schema_version": "mtp-expected-acceptance-v2",
        "state": "PASS",
        "width": width,
        "target": left.get("target"),
        "model_sha256": left.get("model_sha256"),
        "manifest_sha256": identity["manifest_sha256"],
        "source_manifest_sha256": identity["source_manifest_sha256"],
        "support_transform": {"top_k": 20, "top_p": 0.95, "temperature": 1.0},
        "conditions": len(per_prompt),
        "per_prompt": per_prompt,
        "identity": "v2 manifest with referenced v1 Tier A prompts; no fixture copy",
    }


def calculate_single(
    report: Path,
    width: int,
    manifest_path: Path = DEFAULT_MANIFEST,
    draft_vocab_map: Path | None = None,
) -> dict[str, Any]:
    """Derive one current-companion M1 series for the Stage 12 width sweep."""

    width = validate_width(width)
    identity = load_identity(manifest_path)
    document, values = _report_prompt_steps(report, identity, width, None, draft_vocab_map)
    per_prompt = []
    for case_id in sorted(identity["conditions"]):
        row = _find_prompt_entry(document, case_id)
        if row is None or not isinstance(row.get("prompt_sha256"), str):
            raise Stage12IdentityError(f"M1 {case_id} prompt hash is missing")
        draft_rows = row.get("draft_logit_rows")
        target_rows = row.get("target_logit_rows")
        if not isinstance(draft_rows, list) or not isinstance(target_rows, list):
            raise Stage12IdentityError(f"M1 {case_id} raw logits rows are missing")
        counts = {
            f"step{index + 1}": sum(
                isinstance(item, dict) and item.get("block_row") == index
                for item in draft_rows
            )
            for index in range(width)
        }
        bonus_rows = sum(
            isinstance(item, dict) and item.get("block_row") == width
            for item in target_rows
        )
        blocks = row.get("blocks")
        tail = row.get("tail_tokens_omitted")
        if (any(count == 0 for count in counts.values()) or bonus_rows == 0
                or len(draft_rows) != sum(counts.values())
                or isinstance(blocks, bool) or not isinstance(blocks, int) or blocks <= 0
                or any(count != blocks for count in counts.values())
                or bonus_rows != blocks
                or row.get("proposed_draft_tokens") != blocks * width
                or isinstance(tail, bool) or not isinstance(tail, int)
                or not 0 <= tail <= width):
            raise Stage12IdentityError(f"M1 {case_id} proposal/bonus row coverage differs")
        steps = values[case_id]
        per_prompt.append({
            "case_id": case_id,
            "prompt_sha256": row["prompt_sha256"],
            "prompt_token_count": row.get("prompt_token_count"),
            "blocks": blocks,
            "tail_tokens_omitted": tail,
            "proposal_step_row_counts": counts,
            "bonus_target_rows": bonus_rows,
            "target_logits_sha256": row.get("target_logits_sha256"),
            "draft_logits_sha256": row.get("draft_logits_sha256"),
            "by_proposal_step": steps,
            "expected_tokens_per_block": expected_tokens_per_block(steps, width),
        })
    return {
        "schema_version": "mtp-expected-acceptance-width-v2",
        "state": "PASS",
        "width": width,
        "target": document["target"],
        "model_sha256": document.get("model_sha256"),
        "companion_digest": document.get("companion_digest"),
        "draft_vocab_sha256": document.get("draft_vocab_sha256"),
        "manifest_sha256": identity["manifest_sha256"],
        "source_manifest_sha256": identity["source_manifest_sha256"],
        "raw_m1_report_sha256": sha256(report),
        "conditions": len(per_prompt),
        "per_prompt": per_prompt,
    }


def _find_prompt_entry(document: dict[str, Any], case_id: str) -> dict[str, Any] | None:
    rows = document.get("per_prompt")
    if not isinstance(rows, list):
        phase86 = document.get("phase86")
        rows = phase86.get("entries") if isinstance(phase86, dict) else None
    if not isinstance(rows, list):
        return None
    for row in rows:
        if row.get("case_id") == case_id:
            return row
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="single current-companion M1 report")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--width", type=int, required=True, choices=WIDTHS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--baseline-draft-vocab-map", type=Path)
    parser.add_argument("--candidate-draft-vocab-map", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must not already exist")
    if args.report is not None:
        if args.baseline is not None or args.candidate is not None:
            parser.error("--report is exclusive with --baseline/--candidate")
        result = calculate_single(
            args.report, args.width, args.manifest, args.candidate_draft_vocab_map
        )
    else:
        if args.baseline is None or args.candidate is None:
            parser.error("paired mode requires --baseline and --candidate")
        result = calculate(
            args.baseline, args.candidate, args.width, args.manifest,
            args.baseline_draft_vocab_map, args.candidate_draft_vocab_map,
        )
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{result['target']} width={result['width']} conditions={result['conditions']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
