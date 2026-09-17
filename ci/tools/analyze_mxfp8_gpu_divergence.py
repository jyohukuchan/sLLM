#!/usr/bin/env python3
"""Analyze saved teacher-forced logits for the MXFP8 GPU-divergence study.

All source logits hashes and the shared-prefix/report identities are checked
before NumPy maps any logits array. A hash or identity mismatch is fail-closed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TARGETS = {
    "gfx1030": {
        "root": Path(".local-artifacts/mtp-bench/claimb"),
        "series": {"bf16": "fixed2-bf16", "mxfp8": "fixed2-mxfp8", "mxfp6": "fixed2-mxfp6"},
    },
    "gfx1201": {
        "root": Path(".local-artifacts/mtp-bench/claimb-gfx1201"),
        "series": {"bf16": "fixed-bf16", "mxfp8": "fixed-mxfp8", "mxfp6": "fixed-mxfp6"},
    },
}
FORMATS = ("mxfp8", "mxfp6")
MARGIN_BINS = ((0.0, 0.25), (0.25, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, math.inf))
QUANTILES = (0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while True:
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def clean_sha(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower().startswith("sha256:"):
        text = text[7:]
    if len(text) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in text):
        raise ValueError(f"invalid SHA256 value: {value!r}")
    return text.lower()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def load_inputs(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    """Load small JSON metadata and check that every report describes one run."""
    reports: dict[str, Any] = {}
    report_paths: dict[str, str] = {}
    failures: list[str] = []
    for target, config in TARGETS.items():
        for series, dirname in config["series"].items():
            report_path = repo_root / config["root"] / dirname / "report.json"
            report_paths[f"{target}/{series}"] = str(report_path.relative_to(repo_root))
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except Exception as exc:  # keep a clear aggregate preflight result
                failures.append(f"cannot read {report_path}: {exc}")
                continue
            key = f"{target}/{series}"
            reports[key] = report
            if report.get("state") != "PASS":
                failures.append(f"{key}: report state is {report.get('state')!r}, expected 'PASS'")
            if report.get("target") != target:
                failures.append(f"{key}: report target is {report.get('target')!r}, expected {target!r}")
            if report.get("series") != series:
                failures.append(f"{key}: report series is {report.get('series')!r}, expected {series!r}")
            if report.get("benchmark_mode") != "stage0-fixed-prefix-teacher-forcing":
                failures.append(f"{key}: unexpected benchmark_mode {report.get('benchmark_mode')!r}")
            if report.get("schema_version") != "phase85-a16-mtp-stage0-v1":
                failures.append(f"{key}: unexpected schema_version {report.get('schema_version')!r}")
            if report.get("resident_current_bytes_before_shutdown") != 0:
                failures.append(f"{key}: nonzero resident_current_bytes_before_shutdown")
            if not isinstance(report.get("fixed_prefix", {}).get("entries"), list):
                failures.append(f"{key}: fixed_prefix.entries missing")

    if failures:
        raise ValueError("report preflight failed:\n- " + "\n- ".join(failures))

    identity_keys = ("schema_version", "benchmark_mode", "model_sha256")
    common_identity: dict[str, Any] = {}
    for identity_key in identity_keys:
        values = {str(r.get(identity_key)) for r in reports.values()}
        if len(values) != 1:
            failures.append(f"reports disagree on {identity_key}: {sorted(values)}")
        else:
            common_identity[identity_key] = next(iter(values))
    prefix_hashes = {
        clean_sha(r.get("fixed_prefix", {}).get("prefix_file_sha256"))
        for r in reports.values()
    }
    if len(prefix_hashes) != 1:
        failures.append(f"reports disagree on prefix_file_sha256: {sorted(prefix_hashes)}")
    else:
        common_identity["prefix_file_sha256"] = next(iter(prefix_hashes))

    expected_cases: dict[str, dict[str, Any]] | None = None
    logits: list[dict[str, Any]] = []
    for key, report in reports.items():
        entries = report["fixed_prefix"]["entries"]
        cases: dict[str, dict[str, Any]] = {}
        for entry in entries:
            case_id = str(entry.get("case_id", ""))
            if not case_id or case_id in cases:
                failures.append(f"{key}: missing or duplicate case_id {case_id!r}")
                continue
            cases[case_id] = entry
            if entry.get("seed") != 123:
                failures.append(f"{key}/{case_id}: unexpected seed {entry.get('seed')!r}")
            if entry.get("logits_dtype") != "f32-le":
                failures.append(f"{key}/{case_id}: unexpected logits dtype {entry.get('logits_dtype')!r}")
            rows = int(entry.get("row_count", -1))
            vocab = int(entry.get("vocab_size", -1))
            if rows <= 0 or vocab <= 0:
                failures.append(f"{key}/{case_id}: invalid logits shape {rows}x{vocab}")
            if entry.get("nonfinite_value_count") != 0:
                failures.append(f"{key}/{case_id}: report records nonfinite logits")
            logits.append({
                "report_key": key,
                "target": key.split("/", 1)[0],
                "series": key.split("/", 1)[1],
                "case_id": case_id,
                "path": str(entry.get("logits_file", "")),
                "expected_sha256": clean_sha(entry.get("logits_sha256")),
                "expected_bytes": rows * vocab * 4,
                "row_count": rows,
                "vocab_size": vocab,
                "entry": entry,
            })
        if len(cases) != 8:
            failures.append(f"{key}: found {len(cases)} cases, expected 8")
        if expected_cases is None:
            expected_cases = cases
            continue
        if set(cases) != set(expected_cases):
            failures.append(f"{key}: case ids differ from reference report")
        identity_fields = (
            "seed", "prompt_token_count", "output_prefix_token_count", "prompt_sha256",
            "output_prefix_sha256", "full_prefix_sha256", "row_count", "vocab_size", "logits_dtype",
        )
        for case_id in set(cases) & set(expected_cases):
            left, right = expected_cases[case_id], cases[case_id]
            for field in identity_fields:
                if left.get(field) != right.get(field):
                    failures.append(f"{key}/{case_id}: {field} differs from reference report")
            if left.get("row_count") != 256 or left.get("vocab_size") != 248320:
                failures.append(f"{key}/{case_id}: unexpected logits dimensions")

    # All three series and both GPUs must describe the same fixed prefix stream.
    if expected_cases is not None:
        reference_prefix_hashes = {
            clean_sha(expected_cases[c]["full_prefix_sha256"]) for c in expected_cases
        }
        common_identity["case_count"] = len(expected_cases)
        common_identity["full_prefix_sha256_by_case"] = {
            case_id: clean_sha(entry["full_prefix_sha256"])
            for case_id, entry in sorted(expected_cases.items())
        }
        common_identity["distinct_full_prefixes"] = len(reference_prefix_hashes)

    # The path itself must be the file named in its report, under the expected run.
    for item in logits:
        path = Path(item["path"])
        if not path.is_absolute():
            path = repo_root / path
        expected_parent = (
            repo_root
            / TARGETS[item["target"]]["root"]
            / TARGETS[item["target"]]["series"][item["series"]]
            / "logits"
        ).resolve()
        if path.resolve().parent != expected_parent:
            failures.append(f"{item['report_key']}/{item['case_id']}: logits path escapes expected series directory")
        item["path"] = str(path.resolve())

    # Confirm the actual fixed-prefix artifact before any logits arrays are mapped.
    prefix_path = Path(reports["gfx1030/bf16"]["fixed_prefix"]["prefix_file"])
    if not prefix_path.is_absolute():
        prefix_path = repo_root / prefix_path
    try:
        actual_prefix_sha = sha256_file(prefix_path)
        if actual_prefix_sha != common_identity.get("prefix_file_sha256"):
            failures.append(
                f"prefix file SHA256 mismatch: expected {common_identity.get('prefix_file_sha256')}, got {actual_prefix_sha}"
            )
    except Exception as exc:
        actual_prefix_sha = None
        failures.append(f"cannot verify prefix file {prefix_path}: {exc}")
    common_identity["prefix_file"] = str(prefix_path.resolve())
    common_identity["actual_prefix_file_sha256"] = actual_prefix_sha
    common_identity["report_paths"] = report_paths

    if failures:
        raise ValueError("shared report/prefix identity preflight failed:\n- " + "\n- ".join(failures))
    return reports, common_identity, logits


def verify_all_logits(logits: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    """Hash and size-check every referenced logits file before array access."""
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    print(f"hash preflight: checking {len(logits)} logits files", flush=True)
    for index, item in enumerate(logits, 1):
        path = Path(item["path"])
        result = {
            "report_key": item["report_key"],
            "case_id": item["case_id"],
            "path": str(path),
            "expected_sha256": item["expected_sha256"],
            "expected_bytes": item["expected_bytes"],
        }
        try:
            size = path.stat().st_size
            actual_sha = sha256_file(path)
            result.update({"actual_bytes": size, "actual_sha256": actual_sha})
            if size != item["expected_bytes"]:
                errors.append(f"{item['report_key']}/{item['case_id']}: size {size} != expected {item['expected_bytes']}")
            if actual_sha != item["expected_sha256"]:
                errors.append(f"{item['report_key']}/{item['case_id']}: SHA256 mismatch")
        except Exception as exc:
            result["error"] = str(exc)
            errors.append(f"{item['report_key']}/{item['case_id']}: hash failed: {exc}")
        results.append(result)
        if index % 4 == 0 or index == len(logits):
            print(f"hash preflight: {index}/{len(logits)} complete", flush=True)
    manifest = {
        "schema_version": "mxfp8-gpu-divergence-stage-a-hash-v1",
        "state": "FAILED" if errors else "PASS",
        "logits_file_count": len(logits),
        "verified_unique_files": len({item["path"] for item in logits}),
        "results": results,
        "errors": errors,
        "array_reads_started": False,
    }
    write_json(output_dir / "stage-a-hash-verification.json", manifest)
    if errors:
        raise ValueError("logits hash preflight failed; no logits arrays were read:\n- " + "\n- ".join(errors))
    return manifest


def distribution(values: list[float] | np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
    q = np.quantile(array, QUANTILES)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std_population": float(np.std(array)),
        "quantiles": {f"p{int(round(prob * 100)):02d}": float(value) for prob, value in zip(QUANTILES, q)},
    }


def top2_margin(row: np.ndarray, top1_id: int) -> float:
    top_two = np.partition(row, row.size - 2)[-2:]
    second = float(np.min(top_two))
    return float(row[top1_id]) - second


def pair_summary(left: list[bool], right: list[bool]) -> dict[str, Any]:
    left_a = np.asarray(left, dtype=bool)
    right_a = np.asarray(right, dtype=bool)
    both = int(np.count_nonzero(left_a & right_a))
    left_only = int(np.count_nonzero(left_a & ~right_a))
    right_only = int(np.count_nonzero(~left_a & right_a))
    neither = int(np.count_nonzero(~left_a & ~right_a))
    union = both + left_only + right_only
    return {
        "positions": int(left_a.size),
        "both": both,
        "left_only": left_only,
        "right_only": right_only,
        "neither": neither,
        "jaccard": (both / union) if union else None,
    }


def append_prompt_flips(store: dict[tuple[str, str], list[bool]], key: tuple[str, str],
                       values: list[bool]) -> None:
    """Append a prompt's positions so pair summaries span the full run."""
    store.setdefault(key, []).extend(bool(value) for value in values)


def self_test() -> None:
    """Focused two-prompt fixture for pooled flip overlap accounting."""
    pooled: dict[tuple[str, str], list[bool]] = {}
    append_prompt_flips(pooled, ("gfx1030", "mxfp8"), [True] + [False] * 255)
    append_prompt_flips(pooled, ("gfx1201", "mxfp8"), [True] + [False] * 255)
    append_prompt_flips(pooled, ("gfx1030", "mxfp8"), [False, True] + [False] * 254)
    append_prompt_flips(pooled, ("gfx1201", "mxfp8"), [False, False, True] + [False] * 253)
    result = pair_summary(pooled[("gfx1030", "mxfp8")], pooled[("gfx1201", "mxfp8")])
    assert result == {
        "positions": 512, "both": 1, "left_only": 1, "right_only": 1,
        "neither": 509, "jaccard": 1 / 3,
    }, result
    print("Stage A overlap fixture PASS (2 prompts, 512 paired positions)")


def exact_sign_flip_p(values: list[float]) -> float | None:
    """Two-sided exact cluster-level sign-flip p-value, for descriptive context."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return None
    observed = abs(float(np.mean(array)))
    total = 1 << int(array.size)
    extreme = 0
    for mask in range(total):
        signs = np.fromiter((1.0 if mask & (1 << i) else -1.0 for i in range(array.size)), dtype=np.float64)
        if abs(float(np.mean(array * signs))) >= observed - 1e-15:
            extreme += 1
    return extreme / total


def analyze(reports: dict[str, Any], identity: dict[str, Any], logits: list[dict[str, Any]],
            output_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_report = {(item["target"], item["series"], item["case_id"]): item for item in logits}
    metrics: dict[tuple[str, str], dict[str, list[Any]]] = {}
    flip_by_position: dict[tuple[str, str], list[bool]] = {
        (target, fmt): [] for target in TARGETS for fmt in FORMATS
    }
    margins_by_target: dict[str, list[float]] = {target: [] for target in TARGETS}
    records: list[dict[str, Any]] = []
    per_prompt: dict[tuple[str, str], dict[str, int]] = {}
    top1_report_validation = {
        f"{target}/{series}": {"matches": 0, "positions": 0}
        for target in TARGETS for series in ("bf16", *FORMATS)
    }
    current_target = None
    total_positions = 0

    for target in TARGETS:
        current_target = target
        print(f"array analysis: starting {target}", flush=True)
        for case_id in sorted(reports[f"{target}/bf16"]["fixed_prefix"]["entries"][i]["case_id"]
                              for i in range(8)):
            bf_item = by_report[(target, "bf16", case_id)]
            shape = (bf_item["row_count"], bf_item["vocab_size"])
            bf16 = np.memmap(bf_item["path"], dtype="<f4", mode="r", shape=shape)
            candidates = {
                fmt: np.memmap(by_report[(target, fmt, case_id)]["path"], dtype="<f4", mode="r", shape=shape)
                for fmt in FORMATS
            }
            for fmt in FORMATS:
                metrics.setdefault((target, fmt), {
                    "relative_l2": [], "max_abs": [], "abs_top1_delta": [], "top1_delta": [],
                    "flip": [], "margin_at_flip": [],
                })
            m8 = metrics[(target, "mxfp8")]
            m6 = metrics[(target, "mxfp6")]
            prompt_flips = {fmt: [] for fmt in FORMATS}
            prompt_bucket = {"positions": shape[0], "mxfp8_flips": 0, "mxfp6_flips": 0}
            report_top1 = {
                fmt: by_report[(target, fmt, case_id)]["entry"].get("top1")
                for fmt in ("bf16", *FORMATS)
            }
            for fmt, values in report_top1.items():
                if not isinstance(values, list) or len(values) != shape[0]:
                    raise ValueError(f"{target}/{fmt}/{case_id}: report top1 vector has invalid length")
            for row_index in range(shape[0]):
                baseline = np.asarray(bf16[row_index])
                if not np.isfinite(baseline).all():
                    raise ValueError(f"nonfinite BF16 logits at {target}/{case_id}/{row_index}")
                baseline_top1 = int(np.argmax(baseline))
                if int(report_top1["bf16"][row_index]) != baseline_top1:
                    raise ValueError(
                        f"NumPy/report top1 mismatch at {target}/bf16/{case_id}/{row_index}: "
                        f"{baseline_top1} != {report_top1['bf16'][row_index]}"
                    )
                top1_report_validation[f"{target}/bf16"]["matches"] += 1
                top1_report_validation[f"{target}/bf16"]["positions"] += 1
                margin = top2_margin(baseline, baseline_top1)
                margins_by_target[target].append(margin)
                candidate_result: dict[str, dict[str, Any]] = {}
                for fmt, candidate_map in candidates.items():
                    candidate = np.asarray(candidate_map[row_index])
                    if not np.isfinite(candidate).all():
                        raise ValueError(f"nonfinite {fmt} logits at {target}/{case_id}/{row_index}")
                    delta = candidate - baseline
                    rel_l2 = math.sqrt(float(np.sum(delta * delta, dtype=np.float64))) / max(
                        math.sqrt(float(np.sum(baseline * baseline, dtype=np.float64))), np.finfo(np.float64).tiny
                    )
                    max_abs = float(np.max(np.abs(delta)))
                    top1_delta = float(delta[baseline_top1])
                    candidate_top1 = int(np.argmax(candidate))
                    if int(report_top1[fmt][row_index]) != candidate_top1:
                        raise ValueError(
                            f"NumPy/report top1 mismatch at {target}/{fmt}/{case_id}/{row_index}: "
                            f"{candidate_top1} != {report_top1[fmt][row_index]}"
                        )
                    top1_report_validation[f"{target}/{fmt}"]["matches"] += 1
                    top1_report_validation[f"{target}/{fmt}"]["positions"] += 1
                    flipped = candidate_top1 != baseline_top1
                    current = metrics[(target, fmt)]
                    current["relative_l2"].append(rel_l2)
                    current["max_abs"].append(max_abs)
                    current["abs_top1_delta"].append(abs(top1_delta))
                    current["top1_delta"].append(top1_delta)
                    current["flip"].append(flipped)
                    if flipped:
                        current["margin_at_flip"].append(margin)
                        prompt_bucket[f"{fmt}_flips"] += 1
                    prompt_flips[fmt].append(flipped)
                    candidate_result[fmt] = {
                        "top1_id": candidate_top1,
                        "flipped": flipped,
                        "relative_l2": rel_l2,
                        "max_abs": max_abs,
                        "top1_delta": top1_delta,
                    }
                records.append({
                    "target": target,
                    "case_id": case_id,
                    "row": row_index,
                    "bf16_top1_id": baseline_top1,
                    "bf16_top1_top2_margin": margin,
                    "mxfp8_top1_id": candidate_result["mxfp8"]["top1_id"],
                    "mxfp8_flip": int(candidate_result["mxfp8"]["flipped"]),
                    "mxfp8_relative_l2": candidate_result["mxfp8"]["relative_l2"],
                    "mxfp8_max_abs": candidate_result["mxfp8"]["max_abs"],
                    "mxfp8_delta_at_bf16_top1": candidate_result["mxfp8"]["top1_delta"],
                    "mxfp6_top1_id": candidate_result["mxfp6"]["top1_id"],
                    "mxfp6_flip": int(candidate_result["mxfp6"]["flipped"]),
                    "mxfp6_relative_l2": candidate_result["mxfp6"]["relative_l2"],
                    "mxfp6_max_abs": candidate_result["mxfp6"]["max_abs"],
                    "mxfp6_delta_at_bf16_top1": candidate_result["mxfp6"]["top1_delta"],
                })
                total_positions += 1
            for fmt in FORMATS:
                append_prompt_flips(flip_by_position, (target, fmt), prompt_flips[fmt])
            per_prompt[(target, case_id)] = prompt_bucket
            del candidates, bf16
            print(f"array analysis: {target}/{case_id} complete", flush=True)
        print(f"array analysis: finished {target} ({total_positions} rows so far)", flush=True)

    # Rows are keyed in the same case/row order because prefixes were verified.
    expected_positions = sum(
        int(entry["row_count"]) for entry in reports["gfx1030/bf16"]["fixed_prefix"]["entries"]
    )
    cross_gpu: dict[str, Any] = {}
    for fmt in FORMATS:
        left = flip_by_position[("gfx1030", fmt)]
        right = flip_by_position[("gfx1201", fmt)]
        if len(left) != expected_positions or len(right) != expected_positions:
            raise ValueError(
                f"{fmt}: expected {expected_positions} paired positions per GPU, "
                f"got gfx1030={len(left)}, gfx1201={len(right)}"
            )
        cross_gpu[fmt] = pair_summary(left, right)

    within_gpu: dict[str, Any] = {}
    for target in TARGETS:
        within_gpu[target] = pair_summary(
            flip_by_position[(target, "mxfp8")], flip_by_position[(target, "mxfp6")]
        )

    rows_by_target = {
        target: {(row["case_id"], row["row"]): row for row in records if row["target"] == target}
        for target in TARGETS
    }
    baseline_strata: dict[str, dict[str, Any]] = {
        "bf16_top1_agrees_across_gpus": {"positions": 0},
        "bf16_top1_differs_across_gpus": {"positions": 0},
    }
    for fmt in FORMATS:
        for stratum in baseline_strata.values():
            stratum[fmt] = {target: {"flips": 0, "rate": None} for target in TARGETS}
    for key in sorted(set(rows_by_target["gfx1030"]) & set(rows_by_target["gfx1201"])):
        left = rows_by_target["gfx1030"][key]
        right = rows_by_target["gfx1201"][key]
        stratum_name = (
            "bf16_top1_agrees_across_gpus"
            if left["bf16_top1_id"] == right["bf16_top1_id"]
            else "bf16_top1_differs_across_gpus"
        )
        stratum = baseline_strata[stratum_name]
        stratum["positions"] += 1
        for fmt in FORMATS:
            for target, row in (("gfx1030", left), ("gfx1201", right)):
                stratum[fmt][target]["flips"] += int(row[f"{fmt}_flip"])
    for stratum in baseline_strata.values():
        for fmt in FORMATS:
            for target in TARGETS:
                flips = stratum[fmt][target]["flips"]
                stratum[fmt][target]["rate"] = (
                    flips / stratum["positions"] if stratum["positions"] else None
                )

    summaries: dict[str, Any] = {}
    for target in TARGETS:
        summaries[target] = {
            "bf16_margin": distribution(margins_by_target[target]),
            "formats": {},
            "margin_bins": [],
            "prompt_flip_rates": {},
        }
        margins = np.asarray(margins_by_target[target], dtype=np.float64)
        for fmt in FORMATS:
            m = metrics[(target, fmt)]
            flip_mask = np.asarray(m["flip"], dtype=bool)
            summaries[target]["formats"][fmt] = {
                "relative_l2": distribution(m["relative_l2"]),
                "max_abs": distribution(m["max_abs"]),
                "delta_at_bf16_top1": distribution(m["top1_delta"]),
                "absolute_delta_at_bf16_top1": distribution(m["abs_top1_delta"]),
                "top1_flips": int(np.count_nonzero(flip_mask)),
                "top1_agreement": float(1.0 - np.mean(flip_mask)),
                "flip_rate": float(np.mean(flip_mask)),
                "margin_when_flipped": distribution(m["margin_at_flip"]),
            }
        for low, high in MARGIN_BINS:
            bucket_mask = (margins >= low) & (margins < high)
            bucket = {"margin_lo": low, "margin_hi": None if math.isinf(high) else high,
                      "positions": int(np.count_nonzero(bucket_mask))}
            start = 0
            for fmt in FORMATS:
                all_flips = np.asarray(metrics[(target, fmt)]["flip"], dtype=bool)
                bucket_flips = int(np.count_nonzero(all_flips[bucket_mask]))
                bucket[f"{fmt}_flips"] = bucket_flips
                bucket[f"{fmt}_flip_rate"] = (
                    bucket_flips / bucket["positions"] if bucket["positions"] else None
                )
            summaries[target]["margin_bins"].append(bucket)
        for (prompt_target, case_id), counts in per_prompt.items():
            if prompt_target != target:
                continue
            summaries[target]["prompt_flip_rates"][case_id] = {
                fmt: counts[f"{fmt}_flips"] / counts["positions"] for fmt in FORMATS
            }

    prompt_ids = sorted(reports["gfx1030/bf16"]["fixed_prefix"]["entries"][i]["case_id"] for i in range(8))
    prompt_did: list[dict[str, Any]] = []
    did_values: list[float] = []
    for case_id in prompt_ids:
        per_target_advantage = {}
        for target in TARGETS:
            rates = summaries[target]["prompt_flip_rates"][case_id]
            # Positive means MXFP8 preserves more BF16 top-1 choices than MXFP6.
            per_target_advantage[target] = rates["mxfp6"] - rates["mxfp8"]
        did = per_target_advantage["gfx1201"] - per_target_advantage["gfx1030"]
        did_values.append(did)
        prompt_did.append({"case_id": case_id, **per_target_advantage, "gfx1201_minus_gfx1030": did})

    # Pairwise flip set overlap is descriptive: shared flips alone do not establish cause.
    interaction = {
        "definition": "per prompt, (MXFP6 flip rate - MXFP8 flip rate) on gfx1201 minus the same quantity on gfx1030",
        "per_prompt": prompt_did,
        "mean_difference_of_differences": float(np.mean(did_values)),
        "median_difference_of_differences": float(np.median(did_values)),
        "all_prompt_values_positive": bool(all(value > 0 for value in did_values)),
        "exact_two_sided_sign_flip_p_value": exact_sign_flip_p(did_values),
        "inference_note": "Exploratory cluster-level calculation over only 8 fixed prompts; positions within a prompt are not independent and the prompt set is not a random sample.",
    }

    comparison = {}
    for fmt in FORMATS:
        a = summaries["gfx1030"]["formats"][fmt]
        b = summaries["gfx1201"]["formats"][fmt]
        comparison[fmt] = {
            "gfx1201_to_gfx1030_mean_relative_l2_ratio": (
                b["relative_l2"]["mean"] / a["relative_l2"]["mean"]
            ),
            "gfx1201_minus_gfx1030_mean_relative_l2": b["relative_l2"]["mean"] - a["relative_l2"]["mean"],
            "gfx1201_to_gfx1030_mean_max_abs_ratio": b["max_abs"]["mean"] / a["max_abs"]["mean"],
            "gfx1201_minus_gfx1030_flip_rate": b["flip_rate"] - a["flip_rate"],
        }

    summary = {
        "schema_version": "mxfp8-gpu-divergence-stage-a-v1",
        "state": "COMPLETE",
        "source_report_identity": identity,
        "hash_verification": "all 48 report-referenced logits SHA256 and byte sizes passed before NumPy array access",
        "positions_per_target": len(margins_by_target["gfx1030"]),
        "summary_by_target": summaries,
        "numpy_argmax_vs_report_top1": top1_report_validation,
        "continuous_distance_cross_gpu": comparison,
        "paired_flip_sets_across_gpus": cross_gpu,
        "paired_flip_sets_within_gpu": within_gpu,
        "cross_gpu_baseline_top1_stratification": baseline_strata,
        "clustered_flip_advantage_interaction": interaction,
        "interpretation_limits": [
            "GPU-specific BF16 baselines have different target_hidden inputs, so absolute cross-GPU top-1 agreement is not a companion-only comparison.",
            "The distance metrics compare each format with its own GPU's BF16 logits; lower distance describes logit fidelity, not a causal kernel attribution.",
            "Flip-set overlap is descriptive and cannot by itself distinguish shared difficult positions from shared mechanisms.",
            "Conditioning on equal BF16 top-1 IDs across GPUs does not mean the target-hidden vectors or full logits are equal.",
            "The exact sign-flip calculation clusters by the 8 fixed prompts and is exploratory; it does not support broad population inference.",
        ],
        "analyzer": str(Path(__file__).resolve().relative_to(ROOT)),
        "analysis_unix_time": int(time.time()),
    }

    rows_path = output_dir / "stage-a-rows.tsv"
    temporary_rows = rows_path.with_name(rows_path.name + ".tmp")
    fieldnames = list(records[0]) if records else []
    with temporary_rows.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({k: (format(v, ".9g") if isinstance(v, float) else v) for k, v in record.items()})
    os.replace(temporary_rows, rows_path)
    write_json(output_dir / "stage-a-summary.json", summary)
    return summary, records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".local-artifacts/mxfp8-gpu-divergence")
    parser.add_argument("--self-test", action="store_true", help="run the focused two-prompt overlap fixture and exit")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight_path = output_dir / "stage-a-identity-verification.json"
    try:
        reports, identity, logits = load_inputs(repo_root)
        identity_manifest = {
            "schema_version": "mxfp8-gpu-divergence-stage-a-identity-v1",
            "state": "PASS",
            "array_reads_started": False,
            "report_identity": identity,
            "logits_file_count": len(logits),
            "report_referenced_bytes": sum(item["expected_bytes"] for item in logits),
        }
        write_json(preflight_path, identity_manifest)
        hash_manifest = verify_all_logits(logits, output_dir)
        print(
            f"all logits SHA256 verified ({hash_manifest['verified_unique_files']} unique files); beginning array analysis",
            flush=True,
        )
        summary, _ = analyze(reports, identity, logits, output_dir)
    except Exception as exc:
        if not preflight_path.exists():
            write_json(preflight_path, {
                "schema_version": "mxfp8-gpu-divergence-stage-a-identity-v1",
                "state": "FAILED",
                "array_reads_started": False,
                "error": str(exc),
            })
        print(f"Stage A failed closed: {exc}", file=sys.stderr, flush=True)
        return 1

    for target in TARGETS:
        for fmt in FORMATS:
            stat = summary["summary_by_target"][target]["formats"][fmt]
            print(
                f"{target} {fmt}: relL2 mean={stat['relative_l2']['mean']:.9g} "
                f"median={stat['relative_l2']['quantiles']['p50']:.9g}; "
                f"maxAbs mean={stat['max_abs']['mean']:.9g}; "
                f"flip={stat['top1_flips']}/{summary['positions_per_target']} ({stat['flip_rate']:.6%})",
                flush=True,
            )
    print(f"Stage A artifacts: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
