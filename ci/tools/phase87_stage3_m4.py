#!/usr/bin/env python3
"""Derive prompt-paired Stage 3 M4 from fixed M1 and measured M3 blocks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics


REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "ci/matrix/mtp-bench-v1.json"
UUIDS = {"gfx1030": "GPU-76a08c022586fed6", "gfx1201": "GPU-a8e9ddefa2d60f55"}
DRAFT_VOCAB_SHA256 = "24bff6b41785a7729bff183dfea7997e6446173e0df7254cc5761a7519fdebd0"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_tokens_per_block(first: float, second: float) -> float:
    if not 0 <= first <= 1 or not 0 <= second <= 1:
        raise ValueError("M1 proposal-step acceptance must be within [0,1]")
    return 1 + first + first * second


def paired_interval(values: list[float], seed: int = 87, samples: int = 20_000) -> dict:
    if len(values) != 26:
        raise ValueError("M4 requires the 26 Tier A prompt clusters")
    rng = random.Random(seed)
    draws = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(samples))
    return {"mean": statistics.mean(values),
            "ci95": [draws[int(samples * .025)], draws[int(samples * .975)]],
            "clusters": len(values), "resampling_unit": "prompt", "seed": seed,
            "bootstrap_samples": samples}


def read_m3(path: Path, conditions: dict[str, dict]) -> tuple[dict, dict]:
    doc = json.loads(path.read_text())
    if doc.get("state") != "PASS" or not doc.get("cleanup", {}).get("zero"):
        raise ValueError(f"M3 GPU report did not pass and clean up: {path}")
    if doc.get("benchmark_mode") != "stage0-secondary-mtp":
        raise ValueError("M3 report used the wrong benchmark mode")
    target = doc.get("target")
    if target not in UUIDS:
        raise ValueError("M3 target is not an exact tested GPU")
    execution = json.loads((path.parent.parent / "execution.json").read_text())
    jobs = [job for job in execution["jobs"] if job["name"] == path.parent.name]
    if (execution["state"] != "complete" or execution["uuid"] != UUIDS[target]
            or not execution["performance_level_restored"] or len(jobs) != 1
            or jobs[0].get("report_sha256") != sha(path)
            or jobs[0]["env"].get("SLLM_QWEN38_WHOLE_DECODE") != "0"
            or jobs[0]["env"].get("SLLM_PHASE87_STAGE3_M3") != "1"):
        raise ValueError("M3 execution identity or runtime settings differ")
    if target == "gfx1201" and not execution["service_restored"]:
        raise ValueError("R9700 service was not restored")
    suite = doc["secondary_mtp"]
    if suite["warmups"] != 1 or suite["measured"] != 3 or len(suite["entries"]) != 26:
        raise ValueError("M3 report requires 26 cases with 1 warmup and 3 measured")
    suite_path = Path(suite["suite_file"])
    if (not suite_path.is_absolute() or f"sha256:{sha(suite_path)}" != suite["suite_file_sha256"]
            or json.loads(suite_path.read_text())["source_manifest_sha256"] != sha(MANIFEST)):
        raise ValueError("M3 suite file or source manifest differs")
    if doc.get("draft_vocab_sha256") != DRAFT_VOCAB_SHA256:
        raise ValueError("M3 did not use the locked reduced draft vocabulary")
    entries = {}
    for entry in suite["entries"]:
        case = entry["case_id"]
        if case not in conditions or case in entries or entry["seed"] != 123:
            raise ValueError(f"unexpected or repeated M3 case: {case}")
        if entry["prompt_token_count"] != conditions[case]["prompt_tokens_rendered"]:
            raise ValueError(f"M3 prompt token count differs: {case}")
        if entry["output_tokens"] != 256 or entry.get("run") is not None:
            raise ValueError(f"M3 output count or legacy run differs: {case}")
        runs = entry.get("m3_runs")
        if runs is None or [run["sample_kind"] for run in runs] != [
            "warmup", "measured", "measured", "measured"
        ]:
            raise ValueError(f"M3 sample contract differs: {case}")
        values = []
        measured_sha = set()
        for run in runs:
            audit, mtp = run["audit"], run["mtp"]
            memory = run["allocation_after_request_drop"]
            if (audit["selected_backend"] != "hip" or audit["target"] != doc["target"]
                    or audit["fallback_used"] or not audit["all_dispatches_hip"]
                    or audit["kernel_dispatch_count"] <= 0
                    or audit["terminal_logit_non_finite_count"] != 0
                    or mtp["draft_fallback_used"] or not mtp["draft_all_dispatches_hip"]
                    or mtp["proposal_blocks"] <= 0
                    or mtp["fixed_k20_pq_blocks"] != mtp["proposal_blocks"]
                    or mtp["draft_width"] != 2
                    or mtp["committed_output_tokens"] != 256
                    or mtp["proposed_draft_tokens"] <= 0
                    or len(run["generated_tokens"]) != 256
                    or memory["poisoned"] or memory["model_resident"]["current_bytes"] <= 0
                    or memory["request_state"]["current_bytes"] != 0
                    or memory["workspace"]["current_bytes"] != 0):
                raise ValueError(f"M3 execution contract failed: {case}")
            draft_ns = mtp["mtp_decode_proposal_wall_ns"]
            total_ns = run["timing"]["decode_ns"]
            if draft_ns <= 0 or draft_ns >= total_ns:
                raise ValueError(f"M3 time decomposition failed: {case}")
            blocks = mtp["proposal_blocks"]
            values.append({"sample_kind": run["sample_kind"],
                           "draft_ms_per_block": draft_ns / blocks / 1e6,
                           "non_draft_ms_per_block": (total_ns - draft_ns) / blocks / 1e6,
                           "total_ms_per_block": total_ns / blocks / 1e6,
                           "blocks": blocks,
                           "generated_tokens_sha256": run["generated_tokens_sha256"]})
            if run["sample_kind"] == "measured":
                measured_sha.add(run["generated_tokens_sha256"])
        if len(measured_sha) != 1:
            raise ValueError(f"M3 output changed between measured repeats: {case}")
        measured = values[1:]
        entries[case] = {"median": {key: statistics.median(row[key] for row in measured)
                                    for key in ("draft_ms_per_block", "non_draft_ms_per_block",
                                                "total_ms_per_block", "blocks")},
                         "runs": values, "prompt_sha256": entry["prompt_sha256"],
                         "generated_tokens_sha256": measured[0]["generated_tokens_sha256"]}
    if set(entries) != set(conditions):
        raise ValueError("M3 did not cover every Tier A condition")
    doc["execution_binary_sha256"] = execution["binary_sha256"]
    return doc, entries


def calculate(baseline: Path, candidate: Path, m1: Path) -> dict:
    manifest = json.loads(MANIFEST.read_text())
    conditions = {c["cond_id"]: c for c in manifest["conditions"] if c["tier"] == "A"}
    if len(conditions) != 26:
        raise ValueError("mtp-bench-v1 Tier A count changed")
    left, left_entries = read_m3(baseline, conditions)
    right, right_entries = read_m3(candidate, conditions)
    if (left["target"] != right["target"] or left["model_sha256"] != right["model_sha256"]
            or left["execution_binary_sha256"] != right["execution_binary_sha256"]
            or left["secondary_mtp"]["suite_file_sha256"]
            != right["secondary_mtp"]["suite_file_sha256"]
            or left["draft_vocab_sha256"] != right["draft_vocab_sha256"]
            or left["companion_encoding"] is not None
            or right["companion_encoding"] != "nvfp4-w4a4-e2m1-block16-e4m3fn-f32"):
        raise ValueError("M3 control/candidate identity differs")
    acceptance = json.loads(m1.read_text())
    if (acceptance["conditions"] != 26 or acceptance["support_transform"]["top_k"] != 20
            or acceptance["support_transform"]["top_p"] != 0.95
            or acceptance["baseline_draft_vocab_size"] != 98_304
            or acceptance["candidate_draft_vocab_size"] != 98_304):
        raise ValueError("M1 expected-acceptance contract differs")
    m1_baseline = json.loads(Path(acceptance["baseline_report"]).read_text())
    m1_candidate = json.loads(Path(acceptance["candidate_report"]).read_text())
    if (m1_baseline["state"] != "PASS" or m1_candidate["state"] != "PASS"
            or not m1_baseline["cleanup"]["zero"]
            or not m1_candidate["cleanup"]["zero"]
            or not m1_baseline["phase86"]["all_valid"]
            or not m1_candidate["phase86"]["all_valid"]
            or m1_baseline["target"] != left["target"]
            or m1_candidate["target"] != left["target"]
            or m1_baseline["phase86"]["mode"] != "P"
            or m1_candidate["phase86"]["mode"] != "P"
            or m1_baseline["companion_encoding"] is not None
            or m1_candidate["companion_encoding"] != right["companion_encoding"]
            or m1_candidate["companion_digest"] != right["companion_digest"]):
        raise ValueError("M1 source reports do not match M3 target and encodings")
    m1_prompts = {entry["case_id"]: entry["prompt_sha256"]
                  for entry in m1_baseline["phase86"]["entries"]}
    m1_candidate_prompts = {entry["case_id"]: entry["prompt_sha256"]
                            for entry in m1_candidate["phase86"]["entries"]}
    if (set(m1_prompts) != set(conditions)
            or m1_prompts != m1_candidate_prompts
            or any(left_entries[case]["prompt_sha256"] != m1_prompts[case]
                   or right_entries[case]["prompt_sha256"] != m1_prompts[case]
                   for case in conditions)):
        raise ValueError("M1 and M3 rendered prompt token hashes differ")
    rows = acceptance["per_prompt"]
    if {row["case_id"] for row in rows} != set(conditions):
        raise ValueError("M1 prompt set differs from M3")
    paired = []
    for row in rows:
        case = row["case_id"]
        steps = row["by_proposal_step"]
        a = expected_tokens_per_block(steps["step1"]["baseline"],
                                      steps["step2"]["baseline"])
        b = expected_tokens_per_block(steps["step1"]["candidate"],
                                      steps["step2"]["candidate"])
        left_m3 = left_entries[case]["median"]
        right_m3 = right_entries[case]["median"]
        control = 1000 * a / left_m3["total_ms_per_block"]
        changed = 1000 * b / right_m3["total_ms_per_block"]
        paired.append({"case_id": case,
                       "baseline_expected_tokens_per_block": a,
                       "candidate_expected_tokens_per_block": b,
                       "baseline_m3": left_m3, "candidate_m3": right_m3,
                       "baseline_m4_tokens_per_second": control,
                       "candidate_m4_tokens_per_second": changed,
                       "relative_difference": changed / control - 1})
    return {"schema": "phase87-stage3-m4-v1", "target": left["target"],
            "model_sha256": left["model_sha256"],
            "draft_vocab_sha256": left["draft_vocab_sha256"],
            "m3_binary_sha256": left["execution_binary_sha256"],
            "baseline_report_sha256": sha(baseline),
            "candidate_report_sha256": sha(candidate),
            "m1_report_sha256": sha(m1),
            "m1_reference": m1.name,
            "m3_contract": "whole-decode disabled; draft host wall and total decode wall per proposal block, 1 warmup + 3 measured; no CPU fallback",
            "m4_contract": "(1 + step1_M1 + step1_M1*step2_M1) / measured M3 total seconds per block",
            "paired_relative": paired_interval([row["relative_difference"] for row in paired]),
            "per_prompt": paired}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--m1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must not already exist")
    result = calculate(args.baseline, args.candidate, args.m1)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(result["target"], result["m1_reference"], result["paired_relative"])


if __name__ == "__main__":
    main()
