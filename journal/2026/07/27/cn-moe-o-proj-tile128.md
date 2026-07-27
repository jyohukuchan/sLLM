# CN: Qwen3.5-35B-A3B MoE Q-output layout fix and SQ8_0 tile-128 window

## Result

- Took exactly one exclusive R9700 window.  The active AQ4_0 Qwen3.5-9B
  manifest was SHA-256
  `a654d92fe8142fcc0904fe187c96b84c95e0dd18acac61ef25d0cfa6429a08cd`
  before the stop and after recovery; `active.json` was not edited.
- The 35B MoE resident run now loads at the requested 262,144-token/F16-KV
  configuration.  Its sampled process VRAM was `30,371,200,000 B`, versus
  the pre-existing complete ledger `30,858,010,436 B`; the GPU capacity is
  `34,208,743,424 B`.  This is a successful allocation measurement, not an
  OOM result.  The sampled VRAM is `486,810,436 B` below the ledger and leaves
  `3,837,543,424 B` of device VRAM by the process accounting.
- The driver generated 24 actual tokens and independently recomputed the
  raw-BF16 router result for all 40 final-token layers.  All 32 tie-free
  layers matched exactly; the other 8 layers were correctly recorded as
  top-k boundary ties rather than force-passed.  Prompt prefill was
  `10.407136 tok/s` and decode was `11.039290 tok/s`.
- The tile-128 numeric captures completed, unchanged from CM: tile-20 max
  absolute difference `1.3091354375`, tile-128 `2.3758392334`, 471,168 F32
  values each, and no non-finite values.  Existing speed evidence was reused
  as directed; no speed benchmark was rerun.

## MoE Q-output root cause and fix

The generic self-attention bridge classified a gated Q projection only if
`q_rows == 2 * hidden`.  This was an accidental 9B-shaped condition: its
`[8192,4096]` Q matrix makes it true.  The 35B-A3B full-attention layers have
`hidden=2048`, `q_rows=8192`, `16Q`, `head_dim=256`, and a two-channel gated
Q representation, so the same condition is false.  The bridge then treated
Q as 32 plain heads and required O `[2048,8192]`.

`infer_self_attn_q_projection_layout` now uses the complete Q/O contract in
both manifest preflight and resident load.  With O `[hidden,q_heads*value_dim]`,
35B's O `[2048,4096]` unambiguously selects gated 16Q; it cannot be the
plain-32Q interpretation.  The regression test covers both the 35B geometry
and the former 9B geometry.  The rebuilt `rocm-moe-gfx1201` release binary
has SHA-256 `bad1b58c566b3464e1b840b1107be85cebee918dbfac148e919641f7087ac25b`.

The short decoded sample is intentionally retained exactly as generated (the
24-token limit ends it mid-answer): `Thinking Process:\n\n1.  **Analyze the
Request:**\n    *   Task: Reply in one short English`.

## Tile-128 quality capture: direct cause and status

The isolated candidate gateway did start, but the separate Python capture
harness immediately raised `KeyError: 'ULLM_SERVED_MODEL_MANIFEST'`.  The
runner had supplied that variable only to the background gateway process;
the harness also reads it to record the candidate manifest provenance.  This
is the direct cause of the prior readiness/capture failure, not a gateway
startup, GPU, or model-generation failure.

The runner now passes the manifest environment explicitly to the harness and
passes `bash -n`.  However, the failure happened at the end of the one
authorized window; taking a second stop/lock window solely to retry it would
violate CN's one-window instruction.  Therefore no tile-128 candidate text,
including no `javascript_debug_extended` response, exists and tile-128
quality is **unrun**, not pass or fail.  Its stored baseline JavaScript output
must not substitute for candidate evidence: it contains the inaccurate claim
that `NaN` is falsy.

## Reconciliation of the numerical differences

BE's `1.08033e-7` is an isolated, deterministic F32 attention-kernel probe:
one attention output at C=1036, source tile 128, nine partials, compared
with direct and a CPU F32 reference.  CM/CN's two 1--2 scale values compare
three post-512-prefill decode steps of the full SQ8_0 model: each step's
5,120 final-hidden values and 151,936 logits, total 471,168 F32 values per
route.  They are therefore not the same observable and do not contradict.
The full path feeds a changed attention output through all 40 layers and the
sequential SQ8 activation quantizer; at the third decode step the tile-128
final-hidden maximum is `2.375839233` (index 1660), while its corresponding
logits maximum is only `0.377592087`.  The greedy tokens still match in this
three-step microcapture.

The larger tile-128 result does not demonstrate a monotonic merge-count law.
At C=513--515, tile-20 uses 26 partials while tile-128 uses 5, but changing
tile width changes each partial's online-softmax reduction order and its
partial maxima/denominators, not merely how many final merge additions occur.
Those different perturbations then encounter non-linear layer operations and
SQ8 feedback quantization; amplification has no ordering guarantee.  The
captures show this concretely: tile-128 is smaller on final-hidden maxima at
steps 1/2 (`0.422462463`, `0.541427612`) but becomes larger only at step 3
(`2.375839233`), whereas tile-20 is `0.820869446`, `1.309135437`,
`1.302879333`.  Thus the apparent inversion is downstream, sequence- and
content-dependent amplification, not evidence that fewer merge operations
are intrinsically less accurate.

## Recovery

The runner released the R9700 lock before `systemctl start`.  Its immediate
container bridge probe had a transient `container_transport` error, but a
post-recovery retry returned HTTP 200 with `restored` (saved as
`service/restore-response-retry.json`).  The service is active with
`NRestarts=0`; the inactive/disabled `llama-qwen35-udq4.service` was not
started.
