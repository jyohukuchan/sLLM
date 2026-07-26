# Recheck report: rejected projection / format candidates

Date: 2026-07-26.  Scope is `SQ8_0` projection and format candidates only;
decode-attention work belongs to 依頼BH and is not included.

This recheck follows
[`lightweight-promotion-policy-v0.1`](../../../../docs/plans/lightweight-promotion-policy-v0.1.md):
actual generated text is the quality criterion, while exact logits and
top-1 agreement are diagnostic only.  The procedure is deliberately
speed-first: an optimization that does not improve matched full-model decode
throughput stops before a candidate-output capture.

## Outcome

| Candidate | Full-model R9700 result | Generated-text result | Disposition |
| --- | --- | --- | --- |
| `SQ8_0` private handwritten gfx1201 WMMA projection | **9.624875977 tok/s** vs matched CK **14.662647430 tok/s**; ratio 0.656421429 (34.357857% slower) | Not run: the speed-first rule stops a non-faster candidate. | Do not promote. |
| `SQ8_1` W8A8 (K=32 I8 + FP16 scale) | Not measurable on R9700: v0.1 forbids gfx1201 dispatch and no served selector, manifest, or worker exists. | A real code-explanation answer falsely says `NaN` is truthy and `Boolean(NaN)` is `true`. | Do not promote. |
| old handwritten `SQ8_0` attempt-2 | Superseded revision, not a current candidate. | Not rerun. | Do not promote. |
| `SQ9_0` | No packer/reader/oracle/selector/served candidate on gfx1201. | No executable candidate. | Deferred, not measured. |

No candidate was promoted and neither promotion nor rollback tool was invoked.

## `SQ8_0` handwritten WMMA: valid speed-first result

The valid isolated window ran on R9700 (`gfx1201`) from
`2026-07-26T22:35:50+09:00` through `22:59:33+09:00`.  It alternated five CK
and five handwritten runs.  Each run used a 1,028-token prompt, 16 generated
tokens, and counted only feedback decode indices 1--15 (75 tokens per variant
in total).  Model load, prefill, generated index 0, profiler ranges, and GPU
event timing are excluded.  The timing source is the serving binary's
`generated_steps[].synchronized_seconds`.

| Variant | Pooled feedback decode | Pooled latency | Per-run tok/s | Decision |
| --- | ---: | ---: | --- | --- |
| CK control | 14.662647430 tok/s | 68.200507773 ms/token | 14.661099, 14.633736, 14.694418, 14.684115, 14.640062 | control |
| handwritten WMMA | 9.624875977 tok/s | 103.897442667 ms/token | 9.635998, 9.625769, 9.624088, 9.612920, 9.625632 | reject at speed-first |

The prior generally useful baseline is 15.294955751 tok/s (65.381033
ms/token).  This window's CK control is 4.134097% below that historical
number, so the contemporaneous matched CK control -- not the historical
number -- is the qualification comparison.  The WMMA route is nevertheless
also 37.071567% below the historical baseline.

All ten accepted cooldown samples met edge <= 42 C, hotspot <= 45 C, socket
power <= 30 W, and `UNTHROTTLED`.  CK starts were 42/43 C and 13--17 W;
handwritten starts were 42/43 C and 13--14 W.  Raw temperature, clock, power,
and throttle telemetry is retained under
[`sq8_0-handwritten-wmma-speed-first-attempt-3/telemetry`](sq8_0-handwritten-wmma-speed-first-attempt-3/telemetry/).

The candidate is a Qwen3-14B private prototype, not a generic served
candidate for the active Qwen3.5-9B `AQ4_0` service.  In addition to being
slow, it has no eligible manifest/worker that could be passed to the generic
promotion tool.  No candidate-specific promotion route was created.  Because
it was not faster, no fixed-prompt WMMA output capture was run; this is the
explicit speed-first stopping rule, not a numerical rejection.

Compact machine-readable evidence:

- [protocol](sq8_0-handwritten-wmma-speed-first-attempt-3/timing/protocol.txt)
- [CK summary](sq8_0-handwritten-wmma-speed-first-attempt-3/timing/ck-summary.json)
- [WMMA summary](sq8_0-handwritten-wmma-speed-first-attempt-3/timing/handwritten-summary.json)
- [speed decision](sq8_0-handwritten-wmma-speed-first-attempt-3/timing/speed-decision.json)
- [speed-first outcome](sq8_0-handwritten-wmma-speed-first-attempt-3/timing/speed-first-outcome.txt)

## `SQ8_1` W8A8: actual output quality

The fixed ten-case CPU full-model replay is stored in
[`sq8_1-w8a8-cpu-fixed-suite`](sq8_1-w8a8-cpu-fixed-suite/).  It uses the same
source model and rendered prompt as its source-reference control; only the
248 selected projections and their K=32 inputs are fake-quantized.  The
automatic checks found no empty answer, loop, garbling, or total code
abandonment.  That does not make the candidate acceptable.

For `javascript_debug`, the candidate correctly uses `Number.isFinite`, but
then explains that **`NaN` is truthy** and that **`Boolean(NaN)` returns
`true`**.  Both statements are false.  The paired source-reference answer
correctly says that `NaN` is falsy.  A local Node check is preserved in
[`javascript-semantics-check.json`](sq8_1-w8a8-cpu-fixed-suite/javascript-semantics-check.json):
`Boolean(NaN)` is `false`, `Boolean(Infinity)` is `true`, and neither value is
finite.  This is a concrete factual defect in a requested code explanation;
it is not a top-1 or logits threshold decision.

The currently active `AQ4_0` P3 service was also captured with the same
ten prompts, without restarting it:
[`active-aq4_0-fixed-suite`](active-aq4_0-fixed-suite/).  Its
`javascript_debug` answer unfortunately contains the same false NaN-truthiness
claim, so it is **not** a clean semantic control for this individual fact.
That existing active-output defect is recorded rather than concealed.  The
same-model source-reference versus `SQ8_1` comparison is the evidence that
the candidate itself can introduce the error.  The full active-versus-candidate
side-by-side record is
[`active-aq4_0-vs-sq8_1-w8a8`](active-aq4_0-vs-sq8_1-w8a8/).

There is no R9700 `SQ8_1` W8A8 throughput number.  The v0.1 format design
explicitly says not to dispatch W8A8 on gfx1201, and the task prohibits the
historical V620 timing from being reused.  This is reported as
**not measurable**, not as slow or zero throughput.

## Other rejected / registered items found

The inventory is in [candidate-inventory.md](candidate-inventory.md).
gfx1030-only `SQ8_0` fallback work and gfx942/CDNA3 OCP FNUZ prepack work are
not R9700 candidates; neither was measured.  The five manifests in the
served-candidate registry are all unrelated `AQ4_0` manifests: four pass the
static validator and one fails.  Their hashes and exact validator outcomes are
in [registered-candidate-validation.md](registered-candidate-validation.md).

## Service-window and manifest record

Three BJ stop/start windows occurred:

| Window | Time | Status | GPU timing used for conclusion |
| --- | --- | --- | --- |
| Initial pre-measurement abort | 21:45:58--21:45:59 JST | The intentional stop reported `failed`/`MainPID=0`; the old runner rejected that stopped state, exited before GPU isolation, and restored service. | No; zero GPU candidate work. |
| Invalid WMMA invocation | 22:28:40--22:34:29 JST | CK r01 ran, but the prototype rejected WMMA because its required decode-oracle path was omitted.  This incomplete pair is excluded; service restored. | No; no paired WMMA timing. |
| Valid interleaved window | 22:35:50--22:59:33 JST | Five matched CK/WMMA pairs completed; service restored `active/running`. | Yes; the only valid measurement window. |

The runner fix for the required decode-oracle path is commit `1377b0b9`.
Before measuring, BH's foreign exclusive lock was respected; see
[r9700-contention-before-retry.md](r9700-contention-before-retry.md).  No GPU
work, lock release, or service operation was performed by this task while that
foreign lock was held.

The active manifest was initially observed as
`c57a2b6c5827b8ddd102560b3f5efd879711705cf4d8a36f4d7872821d05fca4`.
While waiting for the shared R9700 it changed outside this task to the current
`AQ4_0` P3 manifest:
`a98910dc5bf59dc768e5bcd20bcf58968699540eb1b33df33066dcb6f274fe49`.
The valid window records this latter SHA-256 both before stop and after
restore; this task never wrote `/etc/ullm/served-models/active.json`.
