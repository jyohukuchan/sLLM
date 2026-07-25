# SQ8_1 W8A8 full-model quality gate — pre-measurement criteria

Status: **frozen before the measurement run** (2026-07-26)

This file is the decision contract for this run.  Its numerical thresholds are
not to be loosened after observing a result.  A missing metric, an incomplete
corpus, a non-finite value, or a failed fake-quant harness control is a
No-Go.

Revision note before the gate run: a one-prompt / 16-token **instrument smoke**
was excluded from the gate after its strict control exposed a CPU BF16-versus-
FP32 accumulation difference.  The final gate uses the model converted to
FP32 and the matching FP32 `F.linear` operand boundary, avoiding that mixed-
precision ambiguity before any 20-record gate execution.  No gate threshold
was changed or relaxed; the smoke is not evidence for the decision.

## Scope and reference

- Model: local `Qwen/Qwen3.5-9B`, loaded in FP32 on CPU from its local BF16
  source weights.
- Reference: the unmodified FP32 Hugging Face forward path.
- Primary candidate scope: every one of the 248 transformer `torch.nn.Linear`
  projections selected by the established SQ8_1 collector pattern
  (`self_attn`, `linear_attn`, and `mlp`).  Each selected weight is quantized
  once with `SQ8_1` K=32 signed symmetric int8 and `ceil_fp16` scales.  In
  W8A8, each selected input activation is dynamically quantized by the same
  rule.  Embedding, convolution, norms, non-linearities, and `lm_head` remain
  BF16 because they are not selected by the present SQ8_1 projection path.
- Supplementary stress scope: the same candidate plus the 249th Linear,
  `lm_head`.  It is reported separately, not silently folded into the primary
  deployment-equivalent result.
- Fake-quant compute: codes and stored scales are reconstructed in FP32 and
  consumed by the reference CPU FP32 `F.linear` operand boundary.  This
  validates quantization values and full-model propagation without conflating
  them with a different CPU accumulation kernel; it is not a claim about a HIP
  kernel's accumulation order or throughput.
- Corpus: 20 deterministically evenly spaced records from frozen
  `D_stats-shard-00.jsonl`, truncated at 256 tokens.  The selector must cover
  all five corpus domains with four records each and produce at least 4,000
  valid scored positions.

## Precedents and rationale

The AQ4 P2 gate supplies the required families of checks: logits/hidden
relative L2 and cosine, top-k preservation, BF16 top-1 retention, hidden
maximum error, and token agreement.  Its published AQ4 bounds are loose
because AQ4 is a 4-bit production path (`logits_relative_l2 <= 0.1468`,
`hidden_relative_l2 <= 0.1916`, top-10 overlap >= 0.7900).  SQ8_1 W8A8 is an
8-bit candidate, so this gate intentionally requires substantially tighter
output bounds.

The failed Flash2 staged-body gate used `max_abs <= 2e-5`,
`relative_l2 <= 1e-5`, and cosine >= `0.999999`; those are correct for an
algorithm-preserving implementation substitution, but not for an intentional
int8 quantizer.  They are therefore applied to the BF16 re-expression harness
control below, not to W8A8 itself.  The Flash2 failure values (final-hidden
relative L2 `0.01456836`, logits relative L2 `0.00848364`) are retained as a
warning that standalone numerical success cannot replace a full-model gate.

The only permitted form of non-exact greedy agreement follows the documented
AQ4 decision: a swap can be treated as quantization noise only when it is a
strictly defined near-margin swap to the BF16 runner-up.  The rule is fixed
below; it is not inferred from the observed mismatches.

## Required gates

All gates below apply to the primary 248-projection W8A8 scope unless a row
names another scope.

| Family | Requirement | Pass condition |
| --- | --- | --- |
| Coverage | Frozen corpus and provenance | 20 records, five domains x4, >=4,000 scored positions |
| Validity | Quantization/storage and outputs | all values finite; scales finite/positive; no post-storage clipping; codes in [-127,127] |
| Harness control | FP32 re-expression of each selected Linear | aggregate logits relative L2 <= `1e-5`, aggregate logits max abs <= `2e-5`, final-hidden relative L2 <= `1e-5`, final-hidden max abs <= `2e-5` |
| W8A16 fallback | Weight-only full model | aggregate logits relative L2 <= `0.040`; worst per-prompt logits relative L2 <= `0.060` |
| W8A8 logits | Absolute fidelity | aggregate logits relative L2 <= `0.060`; worst per-prompt logits relative L2 <= `0.080`; aggregate logits max abs <= `1.0`; mean token KL(BF16||W8A8) <= `0.005`; worst per-prompt mean KL <= `0.010` |
| W8A8 versus W8A16 | Incremental activation penalty | W8A8 aggregate logits relative L2 <= both `1.60 * W8A16` and `W8A16 + 0.020` |
| Hidden propagation | Every decoder layer, including final hidden | all finite; maximum layer relative L2 <= `0.080`; final-hidden relative L2 <= `0.060`; final-hidden max abs <= `1.0`; final-hidden relative L2 <= both `1.60 * W8A16` and `W8A16 + 0.020` |
| Top-k | BF16/W8A8 ranking preservation | aggregate top-10 overlap >= `0.950`; BF16 top-1 is in W8A8 top-10 for every scored position |
| Greedy token agreement | Teacher-forced logits at all scored positions | exact agreement >= `99.0%` and 95% Wilson lower bound >= `98.5%`; every mismatch must have W8A8 top-1 equal BF16 top-2 **and** BF16 top1-minus-top2 margin <= `0.050` |
| Supplementary all-Linear stress | 249th `lm_head` added | record the same metrics and status; a failure cannot be used to claim all-Linear W8A8 readiness |

`top-10 overlap` is the set intersection divided by 10.  The top-1 margin is
the raw BF16 logit difference at that position.  Token agreement is
teacher-forced rather than an autoregressive generation claim.

## Outlier attribution decision rule

The run additionally executes an explicitly diagnostic `outlier_bypass_ge4`
candidate: weights remain W8, while only activation K=32 blocks whose
`max(abs(x))/RMS >= 4` bypass activation quantization.  It is an upper bound
for an outlier side route, not a deployable format and not a pass-path.

If base W8A8 fails, this diagnostic is classified as **promising for an
outlier-side route** only when it either satisfies every numeric (non-ranking)
W8A8 gate or removes at least 50% of the W8A8-to-W8A16 aggregate-logit-L2
gap.  Otherwise an outlier side route is not supported by this run.  A
per-channel scale or SmoothQuant design remains unconfirmed until separately
implemented and re-gated; this run may only identify it as a next experiment.

## Decision

`SQ8_1` W8A8 is adoptable for a prequantized projection API only if every
primary-scope gate passes.  A failure fixes W8A16 as the required fallback and
keeps W8A8 explicit-only; it must not be offset by performance measurements.
