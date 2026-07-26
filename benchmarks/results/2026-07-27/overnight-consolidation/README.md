# 2026-07-27 overnight consolidation evidence

This directory records the one-worker AQ4_0 consolidation performed after the
prefill-adaptive-chunk session completed. The frozen runtime build base was
`840a1c7a2fecef6063433b7ffc96b9298840154f`; `main` later advanced to
`95548add4e5c208ee8bf017e5e0ecdea6d95779a` with documentation-only changes.

## Final production identity

- Active manifest SHA-256: `a654d92fe8142fcc0904fe187c96b84c95e0dd18acac61ef25d0cfa6429a08cd`
- Worker: `/opt/ullm/aq4-overnight-consolidation-v0.1/releases/aq4-consolidated-840a1c7a-5a274733/ullm-aq4-worker`
- Worker SHA-256: `5a274733710d9b80a24d34a31ec6a99ac0b2d1e8fcce45904e906926a0e2e903`
- Runtime selection: `AQ4_0`, R9700 (`gfx1201`),
  `aq4_gqa_grouped_split`, `split_tile: 128`, F32 K/V cache default.

## Evidence index

- `candidate-input/` — immutable-manifest and provenance inputs.
- `gpu-window-20260727T075720+0900/` — R9700-only full-model wall-clock
  throughput measurement. `measurement-summary.json` records decode
  `77.8364 tok/s` (two 32-step runs at C=1339) and p2048/M128 prefill
  `975.4217 tok/s`.
- `promotion-20260727T080228+0900/` — generic lightweight promotion evidence:
  active and candidate manifests, readiness records, both 10-prompt outputs,
  comparison, and activation outcome. `comparison.json` reports 10/10 exact
  outputs, no blocking findings, and `passed: true`.

The first local GPU-window invocation stopped before any service mutation: its
guard incorrectly treated a required K/V *kernel* guard as a K/V dtype
selector. `preflight-attempt-1.stderr` preserves that event. The corrected
guard checks only `ULLM_KV_CACHE_DTYPE`, `ULLM_KV_CACHE_TYPE_K`, and
`ULLM_KV_CACHE_TYPE_V`; the successful window is the evidence used here.
