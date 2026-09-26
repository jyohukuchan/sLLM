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
import hashlib
import json
import pathlib
import sys

import numpy as np

VOCAB = 248320
VOCAB_MAP_SCHEMA = "qwen38-draft-vocab-v1"


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"cannot read logits/map file {path}: {exc}") from exc
    return digest.hexdigest()


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a SHA-256 string")
    value = value.removeprefix("sha256:")
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be lowercase SHA-256 hex")
    return value


def _map_metadata_path(path: pathlib.Path) -> pathlib.Path:
    # qwen38_draft_vocab.py accepts an explicit metadata path and the model
    # sidecar uses this spelling. Keep the generator's implicit spelling as a
    # compatibility fallback for locally produced artifacts.
    candidates = (
        path.with_suffix(".metadata.json"),
        path.with_name(path.name + ".metadata.json"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValueError(
        f"candidate draft vocabulary metadata is missing; expected {candidates[0]}"
    )


def load_candidate_draft_vocab_map(path: pathlib.Path) -> np.ndarray:
    """Load and validate a sorted global-token-ID map for candidate logits."""

    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"cannot read candidate draft vocabulary map {path}: {exc}") from exc
    if not payload or len(payload) % 4:
        raise ValueError("candidate draft vocabulary map must be a non-empty u32 payload")
    ids = np.frombuffer(payload, dtype="<u4").copy()
    if np.any(ids[:-1] >= ids[1:]):
        raise ValueError("candidate draft vocabulary map must be strictly increasing")
    if np.any(ids >= VOCAB):
        raise ValueError(
            f"candidate draft vocabulary map contains an out-of-range token ID >= {VOCAB}"
        )

    metadata_path = _map_metadata_path(path)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot parse candidate draft vocabulary metadata {metadata_path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError("candidate draft vocabulary metadata must be an object")
    if metadata.get("schema_version") != VOCAB_MAP_SCHEMA:
        raise ValueError(f"candidate draft vocabulary metadata schema must be {VOCAB_MAP_SCHEMA!r}")
    for key in ("N", "vocab_count"):
        value = metadata.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value != ids.size:
            raise ValueError(f"candidate draft vocabulary metadata {key} must equal map row count")
    if _digest(metadata.get("vocab_sha256"), "candidate draft vocabulary metadata vocab_sha256") != _sha256_file(path):
        raise ValueError("candidate draft vocabulary map SHA-256 does not match metadata")
    for key in ("manifest_sha256", "tokenizer_sha256"):
        _digest(metadata.get(key), f"candidate draft vocabulary metadata {key}")
    special = metadata.get("special_token_ids")
    if not isinstance(special, list) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0 or value >= VOCAB
        for value in special
    ):
        raise ValueError("candidate draft vocabulary metadata special_token_ids is invalid")
    if any(left >= right for left, right in zip(special, special[1:])):
        raise ValueError("candidate draft vocabulary metadata special_token_ids must be sorted")
    if any(not np.any(ids == value) for value in special):
        raise ValueError("candidate draft vocabulary metadata special_token_ids must be selected")
    return ids


def support(
    row: np.ndarray,
    top_k: int,
    top_p: float,
    token_ids: np.ndarray | None = None,
) -> dict[int, float]:
    """Return the selector support, optionally remapped to global token IDs.

    ``row`` always remains the compact candidate head.  The map is used only
    for tie ordering and dictionary keys, so this path never materializes a
    full-vocabulary candidate row.
    """

    if token_ids is None:
        token_ids = np.arange(row.shape[0], dtype=np.int64)
    elif token_ids.shape != row.shape:
        raise ValueError("candidate draft vocabulary map length does not match logits row")
    if row.ndim != 1 or row.size == 0 or not np.isfinite(row).all():
        raise ValueError("draft or target logits row is empty or non-finite")
    k = min(top_k, row.shape[0]) if top_k else row.shape[0]
    # A partition wide enough to survive ties at the k-th value, then an exact
    # (value desc, id asc) order, matching the reference selector.
    width = min(row.shape[0], k + 64)
    idx = np.argpartition(-row, width - 1)[:width]
    order = np.lexsort((token_ids[idx], -row[idx].astype(np.float64)))
    idx = idx[order][:k]
    values = row[idx].astype(np.float64)
    weights = np.exp(values - values[0])
    total = weights.sum()
    cumulative = np.cumsum(weights)
    included = int(np.searchsorted(cumulative, top_p * total, side="left")) + 1
    included = max(1, min(included, k))
    weights = weights[:included]
    weights = weights / weights.sum()
    return dict(zip(token_ids[idx[:included]].tolist(), weights.tolist()))


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


def _validate_logit_rows(rows: object, label: str) -> list[dict]:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{label} must be a non-empty row list")
    keys = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{label} contains a non-object row")
        key = (row.get("sequence_index"), row.get("block_row"))
        if any(isinstance(value, bool) or not isinstance(value, int) for value in key):
            raise ValueError(f"{label} contains a row with invalid sequence_index/block_row")
        if key in keys:
            raise ValueError(f"{label} contains duplicate row {key}")
        keys.add(key)
    return rows


def _validated_memmap(
    entry: dict,
    key: str,
    rows: list[dict],
    width: int,
) -> np.memmap:
    file_value = entry.get(key)
    if not isinstance(file_value, str) or not file_value:
        raise ValueError(f"{key} is missing from report entry")
    path = pathlib.Path(file_value)
    if not path.is_file():
        raise ValueError(f"{key} does not exist: {path}")
    expected_size = len(rows) * width * np.dtype("<f4").itemsize
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"{key} shape mismatch: expected {expected_size} bytes for "
            f"{len(rows)}x{width} f32, got {actual_size}"
        )
    digest_key = key.removesuffix("_file") + "_sha256"
    expected_digest = _digest(entry.get(digest_key), digest_key)
    actual_digest = _sha256_file(path)
    if actual_digest != expected_digest:
        raise ValueError(f"{key} SHA-256 mismatch: expected {expected_digest}, got {actual_digest}")
    return np.memmap(path, dtype="<f4", mode="r", shape=(len(rows), width))


def run_rows(
    entry: dict,
    top_k: int,
    top_p: float,
    candidate_draft_vocab_map: np.ndarray | None = None,
) -> list[tuple[int, float]]:
    draft_rows = entry["draft_logit_rows"]
    target_rows = entry["target_logit_rows"]
    draft_rows = _validate_logit_rows(draft_rows, "draft_logit_rows")
    target_rows = _validate_logit_rows(target_rows, "target_logit_rows")
    draft_width = (
        VOCAB if candidate_draft_vocab_map is None else int(candidate_draft_vocab_map.size)
    )
    draft = _validated_memmap(entry, "draft_logits_file", draft_rows, draft_width)
    target = _validated_memmap(entry, "target_logits_file", target_rows, VOCAB)
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
        q = support(np.asarray(draft[index]), top_k, top_p, candidate_draft_vocab_map)
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
    # A compact candidate row is scored over its own rows, then keyed by the
    # sorted global-token map. This catches accidental local-ID comparison.
    mapped = np.array([10.0, 10.0, 9.0], dtype=np.float32)
    assert list(support(mapped, 2, 1.0, np.array([7, 42, 100], dtype=np.uint32))) == [7, 42]
    try:
        support(np.array([0.0, np.nan], dtype=np.float32), 2, 1.0)
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite logits must fail closed")
    print("self-test PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--baseline", type=pathlib.Path, help="report for the control series")
    parser.add_argument("--candidate", type=pathlib.Path, help="report for the candidate series")
    parser.add_argument(
        "--baseline-draft-vocab-map",
        type=pathlib.Path,
        help="sorted LE-u32 global token IDs for compact baseline draft logits",
    )
    parser.add_argument(
        "--candidate-draft-vocab-map",
        type=pathlib.Path,
        help="sorted LE-u32 global token IDs for compact candidate draft logits",
    )
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
    baseline_map = (
        load_candidate_draft_vocab_map(args.baseline_draft_vocab_map.resolve())
        if args.baseline_draft_vocab_map
        else None
    )
    candidate_map = (
        load_candidate_draft_vocab_map(args.candidate_draft_vocab_map.resolve())
        if args.candidate_draft_vocab_map
        else None
    )
    per_prompt = []
    step_totals = {"baseline": {0: [], 1: []}, "candidate": {0: [], 1: []}}
    for case in sorted(base):
        b = run_rows(base[case], args.top_k, args.top_p, baseline_map)
        c = run_rows(cand[case], args.top_k, args.top_p, candidate_map)
        for name, rows in (("baseline", b), ("candidate", c)):
            for step, value in rows:
                step_totals[name].setdefault(step, []).append(value)
        bm = float(np.mean([v for _, v in b]))
        cm = float(np.mean([v for _, v in c]))
        per_step = {}
        for step in (0, 1):
            baseline_values = [value for row_step, value in b if row_step == step]
            candidate_values = [value for row_step, value in c if row_step == step]
            if not baseline_values or not candidate_values:
                raise ValueError(f"missing proposal step {step + 1} for {case}")
            per_step[f"step{step + 1}"] = {
                "baseline": float(np.mean(baseline_values)),
                "candidate": float(np.mean(candidate_values)),
                "rows_baseline": len(baseline_values),
                "rows_candidate": len(candidate_values),
            }
        per_prompt.append({"case_id": case, "baseline": bm, "candidate": cm,
                           "difference": cm - bm, "baseline_rows": len(b), "candidate_rows": len(c),
                           "by_proposal_step": per_step})
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
        "baseline_draft_vocab_map": (
            str(args.baseline_draft_vocab_map) if args.baseline_draft_vocab_map else None
        ),
        "baseline_draft_vocab_size": int(baseline_map.size) if baseline_map is not None else VOCAB,
        "candidate_draft_vocab_map": (
            str(args.candidate_draft_vocab_map) if args.candidate_draft_vocab_map else None
        ),
        "candidate_draft_vocab_size": int(candidate_map.size) if candidate_map is not None else VOCAB,
        "row_comparison": "prompt means; baseline and candidate row sets may differ",
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
