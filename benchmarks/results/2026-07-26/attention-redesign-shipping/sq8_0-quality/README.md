# `SQ8_0` direct vs GQA-grouped text-quality comparison

This directory is deliberately an **intra-model** comparison.  Both sides use
the Qwen3-14B `SQ8_0` FP8 product, tokenizer, worker binary, source commit,
generation contract, HIP guard set, fixed ten-case prompt suite, and R9700
(`gfx1201`).  It is not an `AQ4_0` versus `SQ8_0` quality comparison.

`direct/manifest.json` has no execution selector.  The gateway therefore
clears any inherited paged-decode experiment selectors before starting the
control worker.  `gqa-grouped-tile20/manifest.json` carries the typed
`worker.execution.paged_decode_attention` contract with
`kernel: gqa_grouped_split` and `split_tile: 20`; manifest mode sets exactly
the grouped/tile/allow-multitile selector set and rejects the pipeline
selector.  The public display names describe the two arms; they do not enter
the worker command or generation request.

The manifests and their validator outputs were produced before the capture;
`build-provenance.json` pins the clean source worktree, built worker SHA-256,
and prompt-suite SHA-256.  The physical captures are
`direct/capture-20260726T160603Z/` and
`gqa-grouped-tile20/capture-20260726T160603Z/`; their comparison is
`comparison-20260726T160603Z/`.  Both workers became ready and completed all
ten requests, with no automated request/empty/repetition/garble/length
finding.  Exact-match rate is 0.000 and is an observation only.

Per the lightweight-promotion policy, the decision is based on actual readable
output and absence of blocking failures rather than exact matching.  The
human-readable audit in `quality-review-20260726T160603Z.md` therefore holds
quality approval: the grouped `python_code` answer provides no Python code,
its JavaScript explanation has a factual error, and its Japanese multiturn
answer is incomplete.  Some control outputs are also limited by the fixed
96/128-token response budgets, so the evidence does not attribute every
incompleteness to attention; it does establish that this candidate must not be
called text-quality-approved.

This is an isolated service-candidate experiment only.  It must not call the
promotion tool or replace the active `AQ4_0` manifest: promoting this `SQ8_0`
manifest would change the product model, which is outside this task.
