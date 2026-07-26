# AQ4_0 P3 deployment evidence

## Scope

This directory records the deployment evaluation for the isolated `AQ4_0` P3 endpoint
`c4c9a9b344fc10e9a77ab0ded3293469d21b2f72`.  It does not advance shared `HEAD` and it does not
include later experimental SQ8_0, AQ5, loader, or Gemma work.

| Item | Value |
|---|---|
| Candidate worker SHA-256 | `ba8c46d6eee81d508f4b2e744ec05d8743a46bf44100ec66257c8d8ae739e265` |
| Candidate manifest SHA-256 | `a98910dc5bf59dc768e5bcd20bcf58968699540eb1b33df33066dcb6f274fe49` |
| Active manifest before evaluation | `c57a2b6c5827b8ddd102560b3f5efd879711705cf4d8a36f4d7872821d05fca4` |
| Product package manifest SHA-256 | `a790a033f57d9c5b9ae0d731a463c26b86aec691f771ce88bb543d676f08e5ad` |
| Device used for direct timings | R9700 / `gfx1201` only |

## Contents

- `source-audit/` — the 47 P3 commits, all AQ4_0-reachable post-P3 changes considered, base
  selection, and Qwen3.5-9B config/layer-pattern confirmation.
- `release-inputs/` — the immutable candidate manifest and build receipt used to stage the fresh
  release under `/opt/ullm/aq4-p3-deployment-v0.1/`.
- `performance/` — direct full-model P3 measurements using the production package/profile.
- `lightweight-promotion-attempt-1/` — fixed prompt-suite evidence for the first generic-tool
  run.  It was interrupted by another session's explicit service stop before candidate mutation.
- `operations/` — service, GPU-lock, manifest-SHA, and StartLimit coordination records.

## Measured P3 performance

The direct candidate timing measured 970.6107 prefill tok/s for 2,048 tokens at chunk width 128
and 73.4568 decode tok/s at C=1339 over 32 measured steps.  The historical values were 982.3835
and 74.29 tok/s respectively.  The relative deltas and thermal/comparability caveats are in
`performance/measurement-summary.md`; the historic 56.6% decode efficiency denominator remains
unconfirmed and is not used as a gate.

## Promotion state

The candidate **was activated** at 22:26:36 JST through the generic lightweight route.  The active
manifest is `a98910dc5bf59dc768e5bcd20bcf58968699540eb1b33df33066dcb6f274fe49`; the worker is
the new P3 build and the gateway is active/running.  The first generic run remained a valuable
fail-closed record: another session stopped the gateway during baseline generation and no bytes
were changed.  After BH released `/run/ullm/r9700.lock` and the StartLimit window was clear, the
fresh second run completed ten real baseline and ten real candidate generations with no blocking
finding.  A read-only rollback preflight remains ready; no rollback was needed.

BJ subsequently used a short isolated SQ8_0 measurement window and temporarily stopped the
gateway without changing the active manifest.  Its restore completed successfully.  Final
post-window confirmation found `/readyz` HTTP 200, the P3 worker executable SHA-256 equal to
`ba8c46d6eee81d508f4b2e744ec05d8743a46bf44100ec66257c8d8ae739e265`, and
`ullm-openai.service` active/running with `NRestarts=0`.
