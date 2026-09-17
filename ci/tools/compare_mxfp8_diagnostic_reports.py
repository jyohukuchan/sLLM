#!/usr/bin/env python3
"""Compare two teacher-forced reports against a BF16 baseline, fail-closed."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def sha_value(value: Any) -> str:
    text = str(value or "").removeprefix("sha256:").lower()
    if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
        raise ValueError(f"invalid SHA256: {value!r}")
    return text


def report(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("state") != "PASS" or not isinstance(data.get("fixed_prefix", {}).get("entries"), list):
        raise ValueError(f"{path}: expected a PASS teacher-forced report with fixed_prefix.entries")
    entries = {}
    for entry in data["fixed_prefix"]["entries"]:
        key = (str(entry.get("case_id", "")), int(entry.get("seed", -1)))
        if not key[0] or key in entries:
            raise ValueError(f"{path}: missing or duplicate (case_id, seed) key {key}")
        entries[key] = entry
    if len(entries) != 8:
        raise ValueError(f"{path}: found {len(entries)} entries, expected 8")
    data["_entries"] = entries
    data["_path"] = str(path.resolve())
    return data


def validate_identity(reports: dict[str, dict[str, Any]]) -> tuple[list[tuple[str, int]], dict[str, Any]]:
    ref = reports["reference"]
    keys = sorted(ref["_entries"])
    problems = []
    for name, current in reports.items():
        if set(current["_entries"]) != set(keys):
            problems.append(f"{name}: (case_id, seed) keys differ")
    for field in ("model_sha256", "benchmark_mode", "schema_version"):
        values = {sha_value(r.get(field)) if field == "model_sha256" else r.get(field) for r in reports.values()}
        if len(values) != 1:
            problems.append(f"reports disagree on {field}")
    prefix_hashes = {sha_value(r.get("fixed_prefix", {}).get("prefix_file_sha256")) for r in reports.values()}
    if len(prefix_hashes) != 1:
        problems.append("reports disagree on fixed-prefix file SHA256")
    if reports["bf16"].get("series") != "bf16":
        problems.append("--bf16 report does not declare series=bf16")
    if ref.get("series") != reports["candidate"].get("series"):
        problems.append("reference and candidate series differ")
    identity_fields = (
        "seed", "prompt_token_count", "output_prefix_token_count", "prompt_sha256",
        "output_prefix_sha256", "full_prefix_sha256", "row_count", "vocab_size", "logits_dtype",
    )
    for key in keys:
        values = [r["_entries"][key] for r in reports.values()]
        for field in identity_fields:
            if any(value.get(field) != values[0].get(field) for value in values[1:]):
                problems.append(f"{key}: reports disagree on {field}")
        if any(int(v.get("row_count", 0)) <= 0 or int(v.get("vocab_size", 0)) <= 0 for v in values):
            problems.append(f"{key}: invalid logits shape")
        if any(v.get("logits_dtype") != "f32-le" for v in values):
            problems.append(f"{key}: logits_dtype must be f32-le")
    if problems:
        raise ValueError("report identity preflight failed: " + "; ".join(problems))

    prefix_files = {}
    for name, current in reports.items():
        if not (raw := current.get("fixed_prefix", {}).get("prefix_file")):
            raise ValueError(f"{name}: fixed-prefix file path missing")
        path = Path(raw)
        if not path.is_absolute():
            path = Path(current["_path"]).parent / path
        path = path.resolve()
        actual = sha256_file(path)
        if actual != sha_value(current["fixed_prefix"]["prefix_file_sha256"]):
            raise ValueError(f"{name}: fixed-prefix file SHA256 mismatch")
        prefix_files[name] = actual
    identity = {
        "model_sha256": next(iter({sha_value(r["model_sha256"]) for r in reports.values()})),
        "fixed_prefix_sha256": next(iter(prefix_hashes)),
        "case_count": len(keys),
        "targets": {name: current.get("target") for name, current in reports.items()},
        "all_reports_same_target": len({r.get("target") for r in reports.values()}) == 1,
        "prefix_file_sha256_verified": prefix_files,
    }
    return keys, identity


def logits_sources(reports: dict[str, dict[str, Any]], keys: list[tuple[str, int]]) -> list[dict[str, Any]]:
    result = []
    for name, current in reports.items():
        for key in keys:
            entry = current["_entries"][key]
            rows, vocab = int(entry["row_count"]), int(entry["vocab_size"])
            path = Path(entry["logits_file"])
            if not path.is_absolute():
                path = Path(current["_path"]).parent / path
            result.append({"report": name, "case_id": key[0], "seed": key[1], "path": str(path.resolve()),
                           "expected_sha256": sha_value(entry["logits_sha256"]),
                           "expected_bytes": rows * vocab * 4, "rows": rows, "vocab": vocab})
    return result


def verify_logits(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cache: dict[str, tuple[str, int]] = {}
    checks = []
    errors = []
    for source in sources:
        path = source["path"]
        try:
            if path not in cache:
                cache[path] = (sha256_file(Path(path)), Path(path).stat().st_size)
            actual_sha, actual_bytes = cache[path]
            check = {k: source[k] for k in ("report", "case_id", "seed", "path", "expected_sha256", "expected_bytes")}
            check.update({"actual_sha256": actual_sha, "actual_bytes": actual_bytes})
            checks.append(check)
            if actual_sha != source["expected_sha256"] or actual_bytes != source["expected_bytes"]:
                errors.append(f"{source['report']}/{source['case_id']}: logits SHA256 or byte-size mismatch")
        except Exception as exc:
            errors.append(f"{source['report']}/{source['case_id']}: logits hash failed: {exc}")
    if errors:
        raise ValueError("logits verification failed; arrays were not read: " + "; ".join(errors))
    return checks


def compare(reports: dict[str, dict[str, Any]], keys: list[tuple[str, int]], identity: dict[str, Any],
            sources: list[dict[str, Any]], output_state: dict[str, Any]) -> dict[str, Any]:
    checks = verify_logits(sources)
    source_by = {(x["report"], x["case_id"], x["seed"]): x for x in sources}
    rows_out = []
    total = {"positions": 0, "reference_candidate_agree": 0, "candidate_bf16_flips": 0, "reference_bf16_flips": 0}
    max_abs_global, relative_l2_sum = 0.0, 0.0
    for key in keys:
        case_id, seed = key
        entries = {name: current["_entries"][key] for name, current in reports.items()}
        rows = int(entries["candidate"]["row_count"])
        vocab = int(entries["candidate"]["vocab_size"])
        top1 = {name: [int(x) for x in entries[name].get("top1", [])] for name in reports}
        if any(len(values) != rows for values in top1.values()):
            raise ValueError(f"{case_id}: report top1 vector length differs from row_count")
        ref_candidate = sum(a == b for a, b in zip(top1["reference"], top1["candidate"]))
        candidate_flips = sum(a != b for a, b in zip(top1["candidate"], top1["bf16"]))
        reference_flips = sum(a != b for a, b in zip(top1["reference"], top1["bf16"]))
        total["positions"] += rows
        total["reference_candidate_agree"] += ref_candidate
        total["candidate_bf16_flips"] += candidate_flips
        total["reference_bf16_flips"] += reference_flips
        hidden = {name: sha_value(entries[name].get("target_hidden_sha256")) for name in reports}
        hidden_equal = {"reference_candidate": hidden["reference"] == hidden["candidate"],
                        "candidate_bf16": hidden["candidate"] == hidden["bf16"],
                        "all_three": len(set(hidden.values())) == 1}
        ref_source = source_by[("reference", case_id, seed)]
        cand_source = source_by[("candidate", case_id, seed)]
        same_logits = ref_source["expected_sha256"] == cand_source["expected_sha256"]
        if same_logits:
            if top1["reference"] != top1["candidate"]:
                raise ValueError(f"{case_id}: identical logits SHA256 but report top1 vectors differ")
            case_max_abs, case_relative_sum = 0.0, 0.0
            distance_state = "BYTE_IDENTICAL_HASH_NO_ARRAY_READ"
        else:
            output_state["array_reads_started"] = True
            shape = (rows, vocab)
            left = np.memmap(ref_source["path"], dtype="<f4", mode="r", shape=shape)
            right = np.memmap(cand_source["path"], dtype="<f4", mode="r", shape=shape)
            case_max_abs, case_relative_sum = 0.0, 0.0
            for index in range(rows):
                a, b = np.asarray(left[index]), np.asarray(right[index])
                if not np.isfinite(a).all() or not np.isfinite(b).all():
                    raise ValueError(f"{case_id}/{index}: nonfinite logits")
                if int(np.argmax(a)) != top1["reference"][index] or int(np.argmax(b)) != top1["candidate"][index]:
                    raise ValueError(f"{case_id}/{index}: NumPy argmax disagrees with report top1")
                delta = b - a
                case_max_abs = max(case_max_abs, float(np.max(np.abs(delta))))
                norm_delta = math.sqrt(float(np.sum(delta * delta, dtype=np.float64)))
                norm_a = math.sqrt(float(np.sum(a * a, dtype=np.float64)))
                case_relative_sum += norm_delta / max(norm_a, np.finfo(np.float64).tiny)
            del left, right
            distance_state = "COMPUTED_NUMPY_ROW_STREAM"
        max_abs_global = max(max_abs_global, case_max_abs)
        relative_l2_sum += case_relative_sum
        rows_out.append({
            "case_id": case_id, "seed": seed, "positions": rows,
            "target_hidden_sha256": hidden, "target_hidden_equal": hidden_equal,
            "top1": {
                "reference_candidate": {"agree": ref_candidate, "changed": rows - ref_candidate, "agreement_rate": ref_candidate / rows},
                "candidate_vs_bf16": {"flips": candidate_flips, "agreement_rate": 1 - candidate_flips / rows},
                "reference_vs_bf16": {"flips": reference_flips, "agreement_rate": 1 - reference_flips / rows},
            },
            "reference_candidate_logits": {
                "byte_identical": same_logits, "distance_state": distance_state,
                "max_abs": case_max_abs, "mean_row_relative_l2": case_relative_sum / rows,
            },
        })
    same_target = identity["all_reports_same_target"]
    all_hidden_equal = all(row["target_hidden_equal"]["all_three"] for row in rows_out)
    return {
        "schema_version": "mxfp8-diagnostic-report-comparison-v1", "state": "COMPLETE",
        "source_reports": {name: current["_path"] for name, current in reports.items()},
        "report_identity": identity,
        "own_gpu_bf16_reference": {
            "candidate_target_matches_bf16": reports["candidate"].get("target") == reports["bf16"].get("target"),
            "reference_target_matches_bf16": reports["reference"].get("target") == reports["bf16"].get("target"),
        },
        "clean_companion_control": bool(same_target and all_hidden_equal),
        "target_hidden_mismatch_cases": [row["case_id"] for row in rows_out if not row["target_hidden_equal"]["all_three"]],
        "logits_verification": {"state": "PASS", "file_references": len(checks), "unique_paths": len({x["path"] for x in checks}), "files": checks},
        "aggregate": {
            **total,
            "reference_candidate_agreement_rate": total["reference_candidate_agree"] / total["positions"],
            "candidate_bf16_flip_rate": total["candidate_bf16_flips"] / total["positions"],
            "reference_bf16_flip_rate": total["reference_bf16_flips"] / total["positions"],
            "reference_candidate_max_abs": max_abs_global,
            "reference_candidate_mean_row_relative_l2": relative_l2_sum / total["positions"],
        },
        "cases": rows_out,
        "caveats": [
            "BF16 flip counts are own-GPU references only when the BF16 report target matches the compared report target.",
            "Different target_hidden_sha256 means the reports do not form a clean companion control, even when the forced prefixes match.",
            "Matching target_hidden hashes establish input identity; they do not by themselves attribute a difference to a particular kernel instruction.",
        ],
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("reference", "candidate", "bf16", "output"):
        parser.add_argument(f"--{option}", required=True, type=Path)
    args = parser.parse_args()
    state = {"array_reads_started": False}
    try:
        reports = {name: report(getattr(args, name).resolve()) for name in ("reference", "candidate", "bf16")}
        keys, identity = validate_identity(reports)
        sources = logits_sources(reports, keys)
        result = compare(reports, keys, identity, sources, state)
    except Exception as exc:
        write_json(args.output, {"schema_version": "mxfp8-diagnostic-report-comparison-v1",
                                 "state": "FAILED", **state, "error": str(exc)})
        print(f"report comparison failed closed: {exc}", file=sys.stderr)
        return 1
    write_json(args.output, result)
    print(f"comparison complete: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
