#!/usr/bin/env python3
"""Derive Stage 12 M4 for one MTP width from v2 M1 and M3 reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
from typing import Any

from phase87_stage12_acceptance import (
    DEFAULT_MANIFEST,
    QWEN38_DRAFT_VOCAB_SHA256,
    QWEN38_NVFP4_COMPANION_DIGEST,
    QWEN38_NVFP4_COMPANION_ENCODING,
    Stage12IdentityError,
    UUIDS,
    expected_tokens_per_block,
    load_identity,
    validate_width,
)


def _valid_sha256(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    digest = value.removeprefix("sha256:")
    return len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)


def sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise Stage12IdentityError(f"cannot read {path}: {error}") from error


def paired_interval(values: list[float], seed: int = 87, samples: int = 20_000) -> dict[str, Any]:
    if len(values) != 26:
        raise Stage12IdentityError("M4 requires the 26 Tier A prompt clusters")
    if samples < 100:
        raise Stage12IdentityError("M4 bootstrap requires at least 100 samples")
    rng = random.Random(seed)
    draws = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(samples))
    return {
        "mean": statistics.mean(values),
        "ci95": [draws[int(samples * 0.025)], draws[int(samples * 0.975)]],
        "clusters": len(values),
        "resampling_unit": "prompt",
        "seed": seed,
        "bootstrap_samples": samples,
    }


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Stage12IdentityError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise Stage12IdentityError(f"{label} must be a JSON object")
    return value


def _validate_m3_header(document: dict[str, Any], identity: dict[str, Any], width: int) -> None:
    if document.get("state") != "PASS":
        raise Stage12IdentityError("M3 report is not PASS")
    cleanup = document.get("cleanup")
    if not isinstance(cleanup, dict) or cleanup.get("zero") is not True:
        raise Stage12IdentityError("M3 report cleanup is not zero")
    if document.get("benchmark_mode") not in (None, "stage0-secondary-mtp", "stage12-secondary-mtp"):
        raise Stage12IdentityError("M3 report used an unsupported benchmark mode")
    target = document.get("target")
    if target not in UUIDS:
        raise Stage12IdentityError("M3 target is not an exact tested GPU")
    if not _valid_sha256(document.get("model_sha256")):
        raise Stage12IdentityError("M3 model identity is missing or invalid")
    if (document.get("companion_encoding") != QWEN38_NVFP4_COMPANION_ENCODING
            or document.get("companion_digest") != QWEN38_NVFP4_COMPANION_DIGEST
            or document.get("draft_vocab_sha256") != QWEN38_DRAFT_VOCAB_SHA256):
        raise Stage12IdentityError("M3 companion or reduced draft vocabulary identity differs")
    for key in ("manifest_sha256", "benchmark_manifest_sha256"):
        if key in document and document[key] != identity["manifest_sha256"]:
            raise Stage12IdentityError(f"M3 report {key} differs from v2 manifest")
    for node in (document, document.get("secondary_mtp", {})):
        if not isinstance(node, dict):
            continue
        for key in ("width", "mtp_width", "draft_width"):
            if key in node and node[key] != width:
                raise Stage12IdentityError(f"M3 report {key} differs from requested width")


def _read_suite_identity(suite: dict[str, Any], identity: dict[str, Any]) -> None:
    suite_file = suite.get("suite_file")
    suite_digest = suite.get("suite_file_sha256")
    if not isinstance(suite_file, str) or not Path(suite_file).is_absolute():
        raise Stage12IdentityError("M3 suite_file must be an absolute path")
    suite_path = Path(suite_file)
    if not suite_path.is_file() or suite_digest != f"sha256:{sha256(suite_path)}":
        raise Stage12IdentityError("M3 suite file hash differs from its report")
    suite_doc = _read_json(suite_path, "M3 suite")
    source_digest = suite_doc.get("source_manifest_sha256")
    if source_digest != identity["manifest_sha256"]:
        raise Stage12IdentityError("M3 suite does not name mtp-bench-v2")


def read_m3(path: Path, identity: dict[str, Any], width: int) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Validate a raw 1-warmup/3-measured M3 report and extract medians."""

    width = validate_width(width)
    document = _read_json(path, f"M3 report {path}")
    _validate_m3_header(document, identity, width)
    suite = document.get("secondary_mtp")
    if not isinstance(suite, dict):
        raise Stage12IdentityError("M3 secondary_mtp section is missing")
    if suite.get("warmups") != 1 or suite.get("measured") != 3:
        raise Stage12IdentityError("M3 requires one warmup and three measured samples")
    _read_suite_identity(suite, identity)
    entries = suite.get("entries")
    if not isinstance(entries, list) or len(entries) != 26:
        raise Stage12IdentityError("M3 must contain exactly 26 Tier A entries")
    expected_cases = set(identity["conditions"])
    result: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise Stage12IdentityError("M3 entry is not an object")
        case_id = entry.get("case_id")
        if case_id not in expected_cases or case_id in result:
            raise Stage12IdentityError(f"unexpected or repeated M3 case: {case_id!r}")
        if (entry.get("prompt_token_count")
                != identity["conditions"][case_id]["prompt_tokens_rendered"]
                or entry.get("output_tokens") != 256
                or entry.get("prompt_sha256") != "sha256:" + identity["frozen_prefixes"][case_id]["prompt_tokens_le_i32_sha256"]):
            raise Stage12IdentityError(f"M3 frozen prompt or output length differs: {case_id}")
        runs = entry.get("m3_runs")
        if not isinstance(runs, list) or [run.get("sample_kind") for run in runs] != [
            "warmup", "measured", "measured", "measured"
        ]:
            raise Stage12IdentityError(f"M3 sample contract differs: {case_id}")
        values = []
        measured_tokens: set[str] = set()
        for run in runs:
            if not isinstance(run, dict):
                raise Stage12IdentityError(f"M3 run is not an object: {case_id}")
            mtp = run.get("mtp", {})
            timing = run.get("timing", {})
            audit = run.get("audit", {})
            if not isinstance(mtp, dict) or not isinstance(timing, dict) or not isinstance(audit, dict):
                raise Stage12IdentityError(f"M3 timing sections are missing: {case_id}")
            if (audit.get("selected_backend") != "hip" or audit.get("target") != document["target"]
                    or audit.get("fallback_used") is not False
                    or audit.get("all_dispatches_hip") is not True
                    or audit.get("submission_count", 0) <= 0
                    or audit.get("kernel_dispatch_count", 0) <= 0
                    or mtp.get("draft_fallback_used") is not False
                    or mtp.get("draft_all_dispatches_hip") is not True):
                raise Stage12IdentityError(f"M3 GPU execution audit differs: {case_id}")
            if mtp.get("draft_width") != width:
                raise Stage12IdentityError(f"M3 draft width differs: {case_id}")
            blocks = mtp.get("proposal_blocks")
            draft_ns = mtp.get("mtp_decode_proposal_wall_ns")
            total_ns = timing.get("decode_ns")
            if (
                isinstance(blocks, bool) or not isinstance(blocks, int) or blocks <= 0
                or isinstance(draft_ns, bool) or not isinstance(draft_ns, (int, float)) or draft_ns <= 0
                or isinstance(total_ns, bool) or not isinstance(total_ns, (int, float))
                or total_ns <= draft_ns
            ):
                raise Stage12IdentityError(f"M3 time decomposition failed: {case_id}")
            token_digest = run.get("generated_tokens_sha256")
            if not _valid_sha256(token_digest):
                raise Stage12IdentityError(f"M3 generated token hash is missing: {case_id}")
            if run.get("sample_kind") == "measured":
                measured_tokens.add(token_digest)
            values.append({
                "sample_kind": run["sample_kind"],
                "draft_ms_per_block": float(draft_ns) / blocks / 1e6,
                "non_draft_ms_per_block": (float(total_ns) - float(draft_ns)) / blocks / 1e6,
                "total_ms_per_block": float(total_ns) / blocks / 1e6,
                "blocks": blocks,
                "generated_tokens_sha256": token_digest,
            })
        if len(measured_tokens) != 1:
            raise Stage12IdentityError(f"M3 output changed between measured repeats: {case_id}")
        measured = values[1:]
        result[case_id] = {
            "median": {
                key: statistics.median(row[key] for row in measured)
                for key in ("draft_ms_per_block", "non_draft_ms_per_block", "total_ms_per_block", "blocks")
            },
            "runs": values,
            "prompt_sha256": entry.get("prompt_sha256"),
            "generated_tokens_sha256": measured[0]["generated_tokens_sha256"],
        }
    if set(result) != expected_cases:
        raise Stage12IdentityError("M3 did not cover every Tier A condition")
    execution_binary = document.get("execution_binary_sha256")
    execution_path = path.parent.parent / "execution.json"
    if not execution_path.is_file():
        raise Stage12IdentityError("M3 exact-GPU execution identity is missing")
    execution = _read_json(execution_path, "M3 execution identity")
    if (execution.get("state") != "complete"
            or execution.get("uuid") != UUIDS[document["target"]]
            or execution.get("performance_level_restored") is not True
            or (document["target"] == "gfx1201" and execution.get("service_restored") is not True)):
        raise Stage12IdentityError("M3 execution identity differs from exact target")
    jobs = execution.get("jobs")
    matching_jobs = [job for job in jobs if job.get("name") == path.parent.name] if isinstance(jobs, list) else []
    if (len(matching_jobs) != 1 or matching_jobs[0].get("exit_code") != 0
            or matching_jobs[0].get("report_sha256") != sha256(path)):
        raise Stage12IdentityError("M3 report differs from the completed GPU job")
    execution_binary = execution.get("binary_sha256")
    if not _valid_sha256(execution_binary):
        raise Stage12IdentityError("M3 execution binary hash is missing")
    document["execution_binary_sha256"] = execution_binary
    return document, result


def _records_from_compact_m3(document: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    rows = document.get("per_prompt")
    if not isinstance(rows, list):
        raise Stage12IdentityError(f"{label} has no per_prompt records")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("case_id"), str):
            raise Stage12IdentityError(f"{label} has an invalid prompt record")
        case_id = row["case_id"]
        if case_id in result or not isinstance(row.get("median"), dict):
            raise Stage12IdentityError(f"{label} has a duplicate or incomplete prompt record: {case_id}")
        median = row["median"]
        total = median.get("total_ms_per_block")
        if isinstance(total, bool) or not isinstance(total, (int, float)) or total <= 0:
            raise Stage12IdentityError(f"{label} has invalid total M3 time: {case_id}")
        result[case_id] = row
    return result


def calculate_from_records(
    m1: dict[str, Any], baseline_m3: dict[str, Any], candidate_m3: dict[str, Any],
    width: int, identity: dict[str, Any],
) -> dict[str, Any]:
    """Pure M4 calculation used by the CLI and focused host tests."""

    width = validate_width(width)
    if m1.get("state") != "PASS" or m1.get("schema_version") != "mtp-expected-acceptance-v2":
        raise Stage12IdentityError("M4 requires a PASS mtp-expected-acceptance-v2 report")
    if m1.get("width") != width or m1.get("manifest_sha256") != identity["manifest_sha256"]:
        raise Stage12IdentityError("M1 width or manifest identity differs from M4")
    for label, report in (("baseline M3", baseline_m3), ("candidate M3", candidate_m3)):
        if report.get("manifest_sha256") and report["manifest_sha256"] != identity["manifest_sha256"]:
            raise Stage12IdentityError(f"{label} manifest identity differs from M4")
        if report.get("target") not in UUIDS:
            raise Stage12IdentityError(f"{label} target is not exact")
    if baseline_m3.get("target") != candidate_m3.get("target") or m1.get("target") != baseline_m3.get("target"):
        raise Stage12IdentityError("M1 and M3 target identities differ")
    models = [m1.get("model_sha256"), baseline_m3.get("model_sha256"), candidate_m3.get("model_sha256")]
    if any(not _valid_sha256(value) for value in models) or len(set(models)) != 1:
        raise Stage12IdentityError("M1 and M3 model identities are missing or differ")
    binaries = [baseline_m3.get("execution_binary_sha256"), candidate_m3.get("execution_binary_sha256")]
    if any(not _valid_sha256(value) for value in binaries) or len(set(binaries)) != 1:
        raise Stage12IdentityError("baseline and candidate M3 binary identities are missing or differ")
    m1_rows = _records_from_compact_m1(m1, width)
    left = _records_from_compact_m3(baseline_m3, "baseline M3")
    right = _records_from_compact_m3(candidate_m3, "candidate M3")
    expected_cases = set(identity["conditions"])
    if set(m1_rows) != expected_cases or set(left) != expected_cases or set(right) != expected_cases:
        raise Stage12IdentityError("M1/M3 prompt set is not the 26 frozen Tier A conditions")
    paired = []
    for case_id in sorted(expected_cases):
        m1_row = m1_rows[case_id]
        expected_prompt_hash = "sha256:" + identity["frozen_prefixes"][case_id]["prompt_tokens_le_i32_sha256"]
        baseline_prompt_hash = left[case_id].get("prompt_sha256")
        candidate_prompt_hash = right[case_id].get("prompt_sha256")
        for side, m3_hash in (("baseline", baseline_prompt_hash), ("candidate", candidate_prompt_hash)):
            m1_hash = m1_row.get(f"prompt_sha256_{side}")
            if m1_hash != expected_prompt_hash or m3_hash != expected_prompt_hash:
                raise Stage12IdentityError(f"M1/M3 frozen prompt identity differs: {case_id} {side}")
        steps = m1_row["by_proposal_step"]
        baseline_steps = {f"step{index}": steps[f"step{index}"]["baseline"] for index in range(1, width + 1)}
        candidate_steps = {f"step{index}": steps[f"step{index}"]["candidate"] for index in range(1, width + 1)}
        base_tokens = expected_tokens_per_block(baseline_steps, width)
        candidate_tokens = expected_tokens_per_block(candidate_steps, width)
        base_m3 = left[case_id]["median"]
        candidate_m3_row = right[case_id]["median"]
        control_speed = 1000.0 * base_tokens / float(base_m3["total_ms_per_block"])
        candidate_speed = 1000.0 * candidate_tokens / float(candidate_m3_row["total_ms_per_block"])
        paired.append({
            "case_id": case_id,
            "baseline_steps": baseline_steps,
            "candidate_steps": candidate_steps,
            "baseline_expected_tokens_per_block": base_tokens,
            "candidate_expected_tokens_per_block": candidate_tokens,
            "baseline_m3": base_m3,
            "candidate_m3": candidate_m3_row,
            "baseline_m4_tokens_per_second": control_speed,
            "candidate_m4_tokens_per_second": candidate_speed,
            "relative_difference": candidate_speed / control_speed - 1.0,
        })
    return {
        "schema_version": "phase87-stage12-m4-v2",
        "state": "PASS",
        "width": width,
        "target": baseline_m3["target"],
        "model_sha256": baseline_m3.get("model_sha256"),
        "manifest_sha256": identity["manifest_sha256"],
        "source_manifest_sha256": identity["source_manifest_sha256"],
        "m1_report_sha256": m1.get("report_sha256"),
        "m4_contract": "(1 + a1 + a1*a2 + ... + cumulative aN) / measured M3 total seconds per block",
        "paired_relative": paired_interval([row["relative_difference"] for row in paired]),
        "conditions": len(paired),
        "per_prompt": paired,
    }


def _records_from_compact_m1(m1: dict[str, Any], width: int) -> dict[str, dict[str, Any]]:
    rows = m1.get("per_prompt")
    if not isinstance(rows, list):
        raise Stage12IdentityError("M1 has no per_prompt records")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("case_id"), str):
            raise Stage12IdentityError("M1 has an invalid prompt record")
        case_id = row["case_id"]
        if case_id in result:
            raise Stage12IdentityError(f"M1 has a duplicate prompt record: {case_id}")
        steps = row.get("by_proposal_step")
        if not isinstance(steps, dict):
            raise Stage12IdentityError(f"M1 has no proposal steps: {case_id}")
        required = {f"step{index}" for index in range(1, width + 1)}
        if set(steps) != required:
            raise Stage12IdentityError(f"M1 proposal steps differ for {case_id}")
        for key in sorted(required):
            value = steps[key]
            if not isinstance(value, dict) or "baseline" not in value or "candidate" not in value:
                raise Stage12IdentityError(f"M1 paired step is incomplete for {case_id}: {key}")
        result[case_id] = row
    return result


def calculate(
    m1_path: Path, baseline_m3_path: Path, candidate_m3_path: Path,
    width: int, manifest_path: Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    identity = load_identity(manifest_path)
    m1 = _read_json(m1_path, f"M1 report {m1_path}")
    baseline_doc, baseline_entries = read_m3(baseline_m3_path, identity, width)
    candidate_doc, candidate_entries = read_m3(candidate_m3_path, identity, width)
    base_records = {
        "state": "PASS", "target": baseline_doc["target"], "model_sha256": baseline_doc.get("model_sha256"),
        "manifest_sha256": identity["manifest_sha256"], "execution_binary_sha256": baseline_doc.get("execution_binary_sha256"),
        "per_prompt": [{"case_id": case, **value} for case, value in baseline_entries.items()],
    }
    candidate_records = {
        "state": "PASS", "target": candidate_doc["target"], "model_sha256": candidate_doc.get("model_sha256"),
        "manifest_sha256": identity["manifest_sha256"], "execution_binary_sha256": candidate_doc.get("execution_binary_sha256"),
        "per_prompt": [{"case_id": case, **value} for case, value in candidate_entries.items()],
    }
    result = calculate_from_records(m1, base_records, candidate_records, width, identity)
    result["m1_report_sha256"] = sha256(m1_path)
    result["baseline_report_sha256"] = sha256(baseline_m3_path)
    result["candidate_report_sha256"] = sha256(candidate_m3_path)
    return result


def calculate_single(
    m1_path: Path,
    m3_path: Path,
    width: int,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> dict[str, Any]:
    """Calculate current-companion M4 for one selected Stage 12 width."""

    width = validate_width(width)
    identity = load_identity(manifest_path)
    m1 = _read_json(m1_path, f"M1 report {m1_path}")
    if (m1.get("schema_version") != "mtp-expected-acceptance-width-v2"
            or m1.get("state") != "PASS"
            or m1.get("width") != width
            or m1.get("manifest_sha256") != identity["manifest_sha256"]):
        raise Stage12IdentityError("single-series M1 identity differs from Stage12 v2")
    m3, m3_rows = read_m3(m3_path, identity, width)
    if (m1.get("target") != m3.get("target")
            or not _valid_sha256(m1.get("model_sha256"))
            or not _valid_sha256(m3.get("model_sha256"))
            or m1.get("model_sha256") != m3.get("model_sha256")):
        raise Stage12IdentityError("single-series M1/M3 target or model identity differs")
    if (m1.get("companion_digest") != QWEN38_NVFP4_COMPANION_DIGEST
            or m1.get("draft_vocab_sha256") != QWEN38_DRAFT_VOCAB_SHA256):
        raise Stage12IdentityError("single-series M1 companion or draft vocabulary identity differs")
    rows = m1.get("per_prompt")
    if not isinstance(rows, list) or len(rows) != 26:
        raise Stage12IdentityError("single-series M1 must contain 26 prompt clusters")
    m1_rows = {row.get("case_id"): row for row in rows if isinstance(row, dict)}
    if set(m1_rows) != set(identity["conditions"]):
        raise Stage12IdentityError("single-series M1 prompt set differs from Tier A")
    per_prompt = []
    for case_id in sorted(identity["conditions"]):
        acceptance = m1_rows[case_id]
        timing = m3_rows[case_id]
        if acceptance.get("prompt_sha256") != timing.get("prompt_sha256"):
            raise Stage12IdentityError(f"single-series M1/M3 prompt hash differs: {case_id}")
        steps = acceptance.get("by_proposal_step")
        expected = expected_tokens_per_block(steps, width)
        measured = timing["median"]
        speed = 1000.0 * expected / float(measured["total_ms_per_block"])
        per_prompt.append({
            "case_id": case_id,
            "by_proposal_step": steps,
            "expected_tokens_per_block": expected,
            "m3_median": measured,
            "m4_tokens_per_second": speed,
        })
    return {
        "schema_version": "phase87-stage12-m4-width-v2",
        "state": "PASS",
        "target": m3["target"],
        "width": width,
        "model_sha256": m3.get("model_sha256"),
        "manifest_sha256": identity["manifest_sha256"],
        "m1_report_sha256": sha256(m1_path),
        "m3_report_sha256": sha256(m3_path),
        "m4_tokens_per_second": paired_interval(
            [row["m4_tokens_per_second"] for row in per_prompt]
        ),
        "conditions": len(per_prompt),
        "per_prompt": per_prompt,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1", type=Path, required=True)
    parser.add_argument("--m3", type=Path, help="single current-companion M3 report")
    parser.add_argument("--baseline-m3", type=Path)
    parser.add_argument("--candidate-m3", type=Path)
    parser.add_argument("--width", type=int, required=True, choices=(2, 3, 4))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must not already exist")
    if args.m3 is not None:
        if args.baseline_m3 is not None or args.candidate_m3 is not None:
            parser.error("--m3 is exclusive with paired M3 inputs")
        result = calculate_single(args.m1, args.m3, args.width, args.manifest)
    else:
        if args.baseline_m3 is None or args.candidate_m3 is None:
            parser.error("paired mode requires --baseline-m3 and --candidate-m3")
        result = calculate(args.m1, args.baseline_m3, args.candidate_m3, args.width, args.manifest)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{result['target']} width={result['width']} conditions={result['conditions']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
