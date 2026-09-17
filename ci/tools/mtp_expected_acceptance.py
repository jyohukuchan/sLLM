#!/usr/bin/env python3
"""Expected speculative acceptance from saved draft/target logits.

For temperature-1 speculative sampling the probability that a draft proposal
is accepted is min(1, p(x)/q(x)) with x ~ q, so the expected acceptance of one
proposal row is sum_t min(p'(t), q'(t)), where p' and q' are the target and
draft distributions after the engine's own support transform. This is the
production acceptance rule itself, it needs no sampling, and it does not
discretise near-tie positions the way top-1 agreement does.

The support transform mirrors the reference token selector
(native/hip/src/token_selector_kernel.hip.cpp, the top_k/top_p branch):
sort logits descending with ties broken by the lower token id, keep top_k,
softmax over those, keep the smallest prefix whose cumulative mass reaches
top_p * sum (at least one candidate), and renormalise.

Input is a teacher-forced report that stores both draft and target logits per
row (the Phase86 P-mode report schema: draft_logit_rows / target_logit_rows with
sequence_index and block_row, plus draft_logits_file / target_logits_file).
Draft and target rows are paired by (sequence_index, block_row) within one run.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

VOCAB = 248320


def support(row: np.ndarray, top_k: int, top_p: float) -> dict[int, float]:
    k = min(top_k, row.shape[0]) if top_k else row.shape[0]
    # A partition wide enough to survive ties at the k-th value, then an exact
    # (value desc, id asc) order, matching the reference selector.
    width = min(row.shape[0], k + 64)
    idx = np.argpartition(-row, width - 1)[:width]
    order = np.lexsort((idx, -row[idx].astype(np.float64)))
    idx = idx[order][:k]
    values = row[idx].astype(np.float64)
    weights = np.exp(values - values[0])
    total = weights.sum()
    cumulative = np.cumsum(weights)
    included = int(np.searchsorted(cumulative, top_p * total, side="left")) + 1
    included = max(1, min(included, k))
    weights = weights[:included]
    weights = weights / weights.sum()
    return dict(zip(idx[:included].tolist(), weights.tolist()))


def expected_acceptance(p: dict[int, float], q: dict[int, float]) -> float:
    return float(sum(min(p.get(token, 0.0), prob) for token, prob in q.items()))


def entries(report: pathlib.Path) -> dict[str, dict]:
    def find(node, key):
        if isinstance(node, dict):
            if key in node:
                return node[key]
            for value in node.values():
                found = find(value, key)
                if found is not None:
                    return found
        elif isinstance(node, list):
            for item in node:
                found = find(item, key)
                if found is not None:
                    return found
        return None

    document = json.loads(report.read_text(encoding="utf-8"))
    return {entry["case_id"]: entry for entry in find(document, "entries")}


def run_rows(entry: dict, top_k: int, top_p: float) -> list[tuple[int, float]]:
    draft_rows = entry["draft_logit_rows"]
    target_rows = entry["target_logit_rows"]
    draft = np.memmap(entry["draft_logits_file"], dtype="<f4", mode="r").reshape(len(draft_rows), VOCAB)
    target = np.memmap(entry["target_logits_file"], dtype="<f4", mode="r").reshape(len(target_rows), VOCAB)
    by_key: dict[tuple[int, int], int] = {}
    for index, row in enumerate(target_rows):
        key = (row["sequence_index"], row["block_row"])
        if key in by_key:
            raise ValueError(f"{entry['case_id']}: duplicate target row {key}")
        by_key[key] = index
    out = []
    for index, row in enumerate(draft_rows):
        key = (row["sequence_index"], row["block_row"])
        if key not in by_key:
            raise ValueError(f"{entry['case_id']}: draft row {key} has no target row")
        q = support(np.asarray(draft[index]), top_k, top_p)
        p = support(np.asarray(target[by_key[key]]), top_k, top_p)
        out.append((int(row["block_row"]), expected_acceptance(p, q)))
    return out


def self_test() -> None:
    # Identical distributions accept with probability one.
    row = np.array([3.0, 1.0, 1.0, -2.0, 0.5], dtype=np.float32)
    s = support(row, 3, 1.0)
    assert abs(expected_acceptance(s, s) - 1.0) < 1e-12
    # Ties at the k boundary keep the lower token id (1 before 2).
    assert list(s) == [0, 1, 2], list(s)
    tie = np.array([1.0, 2.0, 2.0, 2.0], dtype=np.float32)
    assert list(support(tie, 2, 1.0)) == [1, 2]
    # top_p keeps the first candidate that reaches the cutoff, never fewer than one.
    peaked = np.array([10.0, 0.0, 0.0], dtype=np.float32)
    assert list(support(peaked, 3, 0.95)) == [0]
    # With top_k=2 the retained mass is exactly 2, so top_p=0.5 lands exactly on
    # the first candidate (kept alone) and anything above it needs the second.
    both = np.array([0.0, 0.0, -30.0], dtype=np.float32)
    assert list(support(both, 2, 0.5)) == [0]
    assert list(support(both, 2, 0.500001)) == [0, 1]
    # A tiny third candidate kept by top_k still raises the cutoff above the
    # first candidate's mass, as the reference sums every kept candidate first.
    assert list(support(both, 3, 0.5)) == [0, 1]
    # Disjoint supports never accept.
    assert expected_acceptance({0: 1.0}, {1: 1.0}) == 0.0
    # Partial overlap: sum of minima.
    assert abs(expected_acceptance({0: 0.7, 1: 0.3}, {0: 0.4, 1: 0.6}) - 0.7) < 1e-12
    print("self-test PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--baseline", type=pathlib.Path, help="report for the control series")
    parser.add_argument("--candidate", type=pathlib.Path, help="report for the candidate series")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=86)
    parser.add_argument("--out", type=pathlib.Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not (args.baseline and args.candidate and args.out):
        parser.error("--baseline, --candidate and --out are required")

    base, cand = entries(args.baseline), entries(args.candidate)
    if set(base) != set(cand):
        raise SystemExit("baseline and candidate cover different conditions")
    per_prompt = []
    step_totals = {"baseline": {0: [], 1: []}, "candidate": {0: [], 1: []}}
    for case in sorted(base):
        b = run_rows(base[case], args.top_k, args.top_p)
        c = run_rows(cand[case], args.top_k, args.top_p)
        for name, rows in (("baseline", b), ("candidate", c)):
            for step, value in rows:
                step_totals[name].setdefault(step, []).append(value)
        bm = float(np.mean([v for _, v in b]))
        cm = float(np.mean([v for _, v in c]))
        per_prompt.append({"case_id": case, "baseline": bm, "candidate": cm,
                           "difference": cm - bm, "baseline_rows": len(b), "candidate_rows": len(c)})
        print("%-28s %.4f -> %.4f  %+.4f" % (case, bm, cm, cm - bm), flush=True)

    diffs = np.array([row["difference"] for row in per_prompt])
    n = diffs.size
    rng = np.random.default_rng(args.seed)
    boot = np.array([rng.choice(diffs, n, replace=True).mean() for _ in range(args.bootstrap)])
    low, high = np.percentile(boot, [2.5, 97.5])
    observed = abs(diffs.mean())
    flips = sum(abs((rng.choice([-1.0, 1.0], n) * diffs).mean()) >= observed - 1e-15
                for _ in range(args.bootstrap))
    result = {
        "schema_version": "mtp-expected-acceptance-v1",
        "metric": "prompt-mean of per-row sum_t min(p'(t), q'(t)); rows paired by (sequence_index, block_row) within each run",
        "support_transform": {"top_k": args.top_k, "top_p": args.top_p, "temperature": 1.0,
                              "order": "logit desc, lower token id on ties; top_k; softmax; smallest prefix with cumulative >= top_p*sum (>=1); renormalise",
                              "reference": "native/hip/src/token_selector_kernel.hip.cpp top_k/top_p branch"},
        "baseline_report": str(args.baseline), "candidate_report": str(args.candidate),
        "conditions": n,
        "baseline_mean": float(np.mean([r["baseline"] for r in per_prompt])),
        "candidate_mean": float(np.mean([r["candidate"] for r in per_prompt])),
        "mean_difference": float(diffs.mean()),
        "ci95": [float(low), float(high)],
        "signflip_p_two_sided": flips / args.bootstrap,
        "improved_prompts": int((diffs > 0).sum()), "worsened_prompts": int((diffs < 0).sum()),
        "by_proposal_step": {
            f"step{step + 1}": {
                "baseline": float(np.mean(step_totals["baseline"][step])),
                "candidate": float(np.mean(step_totals["candidate"][step])),
                "rows_baseline": len(step_totals["baseline"][step]),
                "rows_candidate": len(step_totals["candidate"][step]),
            } for step in (0, 1) if step_totals["baseline"][step]
        },
        "inference_note": "prompt is the cluster; bootstrap and sign-flip resample prompts, never positions",
        "per_prompt": per_prompt,
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("\nmean %.4f -> %.4f  diff %+.3f pt  95%%CI [%+.3f, %+.3f] pt  p=%.4f  %d/%d" % (
        result["baseline_mean"], result["candidate_mean"], result["mean_difference"] * 100,
        low * 100, high * 100, result["signflip_p_two_sided"],
        result["improved_prompts"], result["worsened_prompts"]))
    for step, values in result["by_proposal_step"].items():
        print("  %s: %.4f -> %.4f  %+.3f pt" % (step, values["baseline"], values["candidate"],
                                               (values["candidate"] - values["baseline"]) * 100))
    return 0


if __name__ == "__main__":
    sys.exit(main())
