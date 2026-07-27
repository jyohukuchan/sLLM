# CM: read-only lightweight harness, then MoE → SQ8_0 tile-128 window

## 前回の要点

- CK and CL each consumed a service window before a usable SQ8_0 quality
  capture: first by pre-creating the runner-owned capture directory, then by
  omitting its required `numeric/` parent.
- The repaired argument contract is an absent `numeric/<route>` target with
  an existing `numeric/` parent.  MoE must run before tile work and must not
  be suppressed by a tile failure.

## 今回の変更点

- Ran `tools/lightweight_promotion.py`'s actual `run_suite` →
  `compare_suites` → `write_comparison_markdown` path against the active
  AQ4_0 Qwen3.5-9B gateway without stopping it.  Two cases produced two
  successful captures, no blocking findings, exact-match rate `1.0`, and
  comparison JSON/Markdown in
  `benchmarks/results/2026-07-27/cm-harness-production-readonly/`.
  `active.json` remained SHA-256
  `a654d92fe8142fcc0904fe187c96b84c95e0dd18acac61ef25d0cfa6429a08cd`.
- Took one R9700 window.  The MoE release binary was the required
  `6ee827e43fa4e4a5e54fd66c1b20eb444e05632245f66349e10cfe409b9e39cd`.
  It reached the resident full-attention loader but stopped at layer 3:
  `self-attn resident o shape mismatch: got [2048,4096] expected [2048,8192]`.
  This is before allocation completion, generation, route read-back, or
  throughput; it is not a VRAM-overflow result.
- The three numeric runners did execute using fresh runner-owned directories:
  direct, grouped tile-20, and grouped tile-128 each emitted their oracle
  captures.  The original summarizer incorrectly prefixed route names twice;
  it now resolves capture paths relative to `numeric/`.  The completed
  capture comparison (471,168 F32 values per route, no non-finite values) is
  `1.3091354375` max absolute difference for tile-20 and
  `2.3758392334` for tile-128.  These are observations only, not a quality
  gate.
- The isolated tile-128 gateway capture exited before readiness/case capture.
  Its stdout/stderr were not retained by the old runner, so its immediate
  cause is **unconfirmed**.  No candidate text, comparison Markdown, or
  `javascript_debug_extended` response exists; tile-128 quality is therefore
  **not run**, not failed or passed.  The runner now persists both capture
  and comparison harness stdout/stderr for a future diagnostic.
- The runner no longer uses `set -e`; MoE failure did not prevent all three
  numeric routes from running.  Production was started only after the lock
  was released.  The script's immediate container restore probe was a
  transient `container_transport` failure, but the subsequent direct
  OpenWebUI bridge completion returned HTTP 200 / `restored` with
  `ActiveState=active` and `NRestarts=0`.

## 次の行動

- Fix the Qwen3.5-35B-A3B full-attention Q projection layout admission.  At
  layer 3, the source has hidden size 2048, q-proj `[8192,2048]`, K/V
  `[512,2048]`, and o-proj `[2048,4096]`; the generic gated-Q inference is
  treating the 8192 Q rows as 32 plain heads rather than 16 gated heads.
  Rebuild and independently review the release before another authorized
  window.
- Diagnose the isolated tile gateway's pre-readiness failure using the newly
  persisted harness stderr.  A future window must run the eight case suite
  before drawing any tile-128 text-quality conclusion; the numeric deltas do
  not replace that requirement.
