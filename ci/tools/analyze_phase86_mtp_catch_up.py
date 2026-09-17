#!/usr/bin/env python3
"""Summarize Phase86 free generation and prompt-cluster paired observations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics


def paired_cluster(values: list[float], *, seed: int = 86, samples: int = 20000) -> dict:
    """Resample whole prompt differences, never individual token positions."""
    if len(values) < 2:
        raise ValueError("at least two prompt clusters required")
    rng = random.Random(seed)
    mean = statistics.mean(values)
    draws = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(samples))
    # Two-sided random sign test under exchangeability of the paired labels.
    extreme = sum(abs(statistics.mean(v * rng.choice((-1, 1)) for v in values))
                  >= abs(mean) - 1e-15 for _ in range(samples))
    return {"prompt_clusters": len(values), "mean_difference": mean,
            "paired_bootstrap_95pct": [draws[int(samples * .025)], draws[int(samples * .975)]],
            "sign_flip_two_sided_p": (extreme + 1) / (samples + 1),
            "resampling_unit": "prompt", "seed": seed, "resamples": samples,
            "prompt_differences": values}


def free_report(path: Path) -> dict:
    doc = json.loads(path.read_text())
    if doc["state"] != "PASS" or not doc["cleanup"]["zero"]:
        raise ValueError(f"invalid GPU evidence: {path}")
    if (doc["sampling"]["mode"] != "gpu_fixed" or doc["sampling"]["replay_inputs"]
            or not doc["mtp"]["requested"] or doc["mtp"]["draft_width"] != 2
            or doc["mtp"]["execution"] != "fixed_gpu_sampler_speculative"):
        raise ValueError("M3 requires width-two fixed GPU MTP free generation")
    execution = json.loads((path.parent.parent / "execution.json").read_text())
    jobs = [j for j in execution["jobs"] if j["name"] == path.parent.name]
    if len(jobs) != 1 or jobs[0].get("report_sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("free report is not bound to its execution record")
    catch_up = jobs[0]["env"].get("SLLM_QWEN_MTP_CATCH_UP", "off")
    rows = []
    for row in doc["rows"]:
        runs = []
        for run in row["runs"]:
            if run["sample_kind"] != "measured":
                continue
            audit, mtp = run["audit"], run["mtp"]
            if (audit["selected_backend"] != "hip" or not audit["all_dispatches_hip"]
                    or audit["fallback_used"] or audit["kernel_dispatch_count"] <= 0
                    or audit["terminal_logit_non_finite_count"] != 0
                    or not mtp["draft_all_dispatches_hip"] or mtp["draft_fallback_used"]
                    or mtp["draft_kernel_dispatch_count"] <= 0):
                raise ValueError(f"dispatch evidence failed: {path}")
            blocks = mtp["proposal_blocks"]
            draft_ns = mtp["mtp_decode_proposal_wall_ns"]
            decode_ns = run["timing"]["decode_ns"]
            if blocks <= 0 or decode_ns < draft_ns:
                raise ValueError("invalid time decomposition")
            draft_ms = draft_ns / blocks / 1e6
            other_ms = (decode_ns - draft_ns) / blocks / 1e6
            tokens = mtp["committed_decode_tokens"]
            # Use the observed committed count: terminal truncation can make
            # blocks + accepted differ by one from the actual decode count.
            derived = tokens / (blocks * (draft_ms + other_ms) / 1000)
            runs.append({"blocks": blocks, "accepted": mtp["accepted_draft_tokens"],
                         "proposed": mtp["proposed_draft_tokens"], "decode_tokens": tokens,
                         "draft_ms_per_block": draft_ms, "non_draft_ms_per_block": other_ms,
                         "derived_decode_tokens_per_second": derived,
                         "generated_tokens_sha256": run["generated_tokens_sha256"]})
        if not runs:
            raise ValueError("no measured runs")
        rows.append({"prompt_tokens": row["prompt_tokens"], "output_tokens": row["output_tokens"],
                     "runs": runs, "median": {k: statistics.median(r[k] for r in runs) for k in
                     ("blocks", "accepted", "proposed", "decode_tokens", "draft_ms_per_block",
                      "non_draft_ms_per_block", "derived_decode_tokens_per_second")},
                     "deterministic_generated_tokens": row["deterministic_generated_tokens"],
                     "generated_tokens_sha256": row["generated_tokens_sha256"]})
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "target": doc["target"], "cleanup_zero": True, "rows": rows,
            "catch_up": catch_up, "binary_sha256": execution["binary_sha256"],
            "sampling": doc["sampling"], "mtp": doc["mtp"], "fixture": doc["fixture"],
            "protocol": doc["protocol"], "model": doc["model"]}


def forced_report(path: Path) -> tuple[dict, dict[str, dict]]:
    doc = json.loads(path.read_text())
    report = doc["phase86"]
    if doc["state"] != "PASS" or not doc["cleanup"]["zero"] or not report["all_valid"]:
        raise ValueError(f"invalid forced evidence: {path}")
    entries = {e["case_id"]: e for e in report["entries"]}
    if not entries or len(entries) != len(report["entries"]):
        raise ValueError("empty or duplicate prompt entries")
    for entry in entries.values():
        if not entry["blocks"] or not entry["target_logit_rows"] or not entry["draft_logit_rows"]:
            raise ValueError("prefill dispatch alone is not a forced measurement")
        for audit in entry["audit"].values():
            if (audit["selected_backend"] != "hip" or audit["target"] != doc["target"]
                    or audit["fallback_used"] or not audit["all_dispatches_hip"]
                    or audit["kernel_dispatch_count"] <= 0):
                raise ValueError(f"invalid forced dispatch: {path}")
    return doc, entries


def forced_pair(control_path: Path, candidate_path: Path) -> dict:
    control_doc, controls = forced_report(control_path)
    candidate_doc, candidates = forced_report(candidate_path)
    if control_doc["target"] != candidate_doc["target"] or controls.keys() != candidates.keys():
        raise ValueError("forced comparison must share GPU and prompt clusters")
    if control_doc["phase86"]["mode"] != "P" or candidate_doc["phase86"]["mode"] != "P":
        raise ValueError("acceptance comparison requires P modes")
    if (control_doc["phase86"]["catch_up"] != "off"
            or candidate_doc["phase86"]["catch_up"] != "separate"):
        raise ValueError("forced pair requires off control and separate candidate")
    prompts = []
    bands = {name: {"pairs": 0, "top1_changed": 0, "forced_match_delta": 0}
             for name in ("[0,.25)", "[.25,1)", "[1,4)", "[4,inf)")}
    for name in sorted(controls):
        a, b = controls[name], candidates[name]
        if a["full_prefix_sha256"] != b["full_prefix_sha256"]:
            raise ValueError(f"forced token path changed: {name}")
        rates = [entry["accepted_draft_tokens"] / entry["proposed_draft_tokens"]
                 for entry in (a, b)]
        # P's own accepted prefix determines its next block start. Match only
        # equal absolute input position AND proposal step; report the coverage.
        maps = [{(r["sequence_index"], r["block_row"]): r
                 for r in entry["draft_logit_rows"]} for entry in (a, b)]
        common = maps[0].keys() & maps[1].keys()
        changed = 0
        for key in common:
            left, right = maps[0][key], maps[1][key]
            margin = left["margin"]
            band = "[0,.25)" if margin < .25 else "[.25,1)" if margin < 1 else "[1,4)" if margin < 4 else "[4,inf)"
            flip = left["top1_token"] != right["top1_token"]
            changed += flip
            bands[band]["pairs"] += 1
            bands[band]["top1_changed"] += flip
            bands[band]["forced_match_delta"] += int(right["top1_matches_forced"]) - int(left["top1_matches_forced"])
        prompts.append({"case_id": name, "control_acceptance": rates[0],
                        "candidate_acceptance": rates[1], "difference": rates[1] - rates[0],
                        "control_blocks": a["blocks"], "candidate_blocks": b["blocks"],
                        "control_proposed": a["proposed_draft_tokens"],
                        "candidate_proposed": b["proposed_draft_tokens"],
                        "control_tokens_per_block": 1 + a["accepted_draft_tokens"] / a["blocks"],
                        "candidate_tokens_per_block": 1 + b["accepted_draft_tokens"] / b["blocks"],
                        "matched_position_step_pairs": len(common), "matched_top1_changed": changed,
                        "control_tail_omitted": a["tail_tokens_omitted"],
                        "candidate_tail_omitted": b["tail_tokens_omitted"]})
    return {"target": control_doc["target"], "control": str(control_path),
            "candidate": str(candidate_path),
            "contract": "forced-column top1 acceptance; not production p/q acceptance",
            "schedule_limit": "candidate-dependent block boundaries; margin pairs use only shared position/step",
            "prompts": prompts, "paired": paired_cluster([p["difference"] for p in prompts]),
            "margin_bands_control_draft": bands}


def derived_pair(forced: dict, control_path: Path, candidate_path: Path) -> dict:
    a, b = free_report(control_path), free_report(candidate_path)
    if a["target"] != b["target"] or a["target"] != forced["target"]:
        raise ValueError("M1 and M3 must share the same exact target")
    if a["catch_up"] != "off" or b["catch_up"] != "separate":
        raise ValueError("timing pair requires off control and separate candidate")
    for key in ("sampling", "mtp", "fixture", "protocol", "model", "binary_sha256"):
        if a[key] != b[key]:
            raise ValueError(f"free comparison changed {key}")
    if [(r["prompt_tokens"], r["output_tokens"]) for r in a["rows"]] != [
            (r["prompt_tokens"], r["output_tokens"]) for r in b["rows"]]:
        raise ValueError("free comparison changed prompt/output fixture size")
    costs = []
    for report in (a, b):
        if len(report["rows"]) != 1:
            raise ValueError("one free-generation timing fixture required")
        row = report["rows"][0]
        costs.append(statistics.median(r["draft_ms_per_block"] + r["non_draft_ms_per_block"]
                                       for r in row["runs"]) / 1000)
    prompts = []
    for p in forced["prompts"]:
        control = p["control_tokens_per_block"] / costs[0]
        candidate = p["candidate_tokens_per_block"] / costs[1]
        prompts.append({"case_id": p["case_id"], "control_tps": control,
                        "candidate_tps": candidate, "difference": candidate - control,
                        "relative_difference": candidate / control - 1})
    return {"target": forced["target"], "control_timing": a, "candidate_timing": b,
            "seconds_per_block": costs, "prompts": prompts,
            "paired_absolute": paired_cluster([p["difference"] for p in prompts]),
            "paired_relative": paired_cluster([p["relative_difference"] for p in prompts]),
            "scope": "M1 forced top1 tokens/block with M3 fixed-GPU p/q block cost; diagnostic estimate",
            "limitations": "prompt bootstrap conditional on observed median M3 cost; not a p/q throughput guarantee"}


def target_conditioned_comparison(reference_path: Path, production_path: Path) -> dict:
    tdoc, refs = forced_report(reference_path)
    pdoc, cases = forced_report(production_path)
    if (tdoc["target"] != pdoc["target"] or refs.keys() != cases.keys()
            or tdoc["phase86"]["mode"] != "T" or pdoc["phase86"]["mode"] != "P"):
        raise ValueError("T/P comparison needs matching targets and prompt sets")
    prompts = []
    for name, p in sorted(cases.items()):
        t = refs[name]
        if t["full_prefix_sha256"] != p["full_prefix_sha256"]:
            raise ValueError("T/P forced tokens differ")
        reference = {r["sequence_index"]: r for r in t["draft_logit_rows"]}
        target_rows = {(r["sequence_index"], r["block_row"]): r for r in p["target_logit_rows"]}
        steps = []
        for step in (0, 1):
            rows = [r for r in p["draft_logit_rows"] if r["block_row"] == step]
            pairs = [(reference[r["sequence_index"]], r) for r in rows]
            steps.append({"step": step + 1, "pairs": len(pairs),
                          "top1_equal_to_T": sum(a["top1_token"] == b["top1_token"] for a, b in pairs),
                          "T_margin_median": statistics.median(a["margin"] for a, _ in pairs),
                          "P_margin_median": statistics.median(b["margin"] for _, b in pairs),
                          "P_matches_target_top1": sum(r["top1_token"] == target_rows[
                              (r["sequence_index"], step)]["top1_token"] for r in rows)})
        prompts.append({"case_id": name, "steps": steps,
                        "T_target_hidden_sha256": t["target_hidden_sha256"],
                        "P_target_hidden_sha256": p["target_hidden_sha256"]})
    return {"target": tdoc["target"], "reference": str(reference_path),
            "production": str(production_path), "prompts": prompts,
            "limitation": "T uses M1 target transitions and P uses M3 verify/replay; this is not a pure hidden-conditioning causal contrast"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--free", type=Path, action="append", default=[])
    parser.add_argument("--paired-prompts", type=Path,
                        help="JSON array of per-prompt candidate minus control differences")
    parser.add_argument("--forced-pair", type=Path, nargs=2, action="append", default=[])
    parser.add_argument("--timing-pair", type=Path, nargs=2, action="append", default=[],
                        help="free reports corresponding in order to each forced pair")
    parser.add_argument("--target-reference", type=Path, nargs=2, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"schema": "phase86-analysis-v1", "free_generation": [free_report(p) for p in args.free]}
    report["forced_pairs"] = [forced_pair(a, b) for a, b in args.forced_pair]
    if args.timing_pair:
        if len(args.timing_pair) != len(report["forced_pairs"]):
            raise ValueError("each forced pair needs its timing pair")
        report["derived_pairs"] = [derived_pair(f, *t) for f, t in
                                   zip(report["forced_pairs"], args.timing_pair)]
    report["target_conditioned_comparisons"] = [target_conditioned_comparison(a, b)
                                               for a, b in args.target_reference]
    if args.paired_prompts:
        report["paired"] = paired_cluster(json.loads(args.paired_prompts.read_text()))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
