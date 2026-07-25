# SQ8_0 paged-decode source-tile numerical fix

## Verdict

**The frozen full-model numerical gate passes for tile 128 and tile 256, but
the multi-tile optimization is not rehabilitated and direct remains the
default.** The tile experiment now uses its existing split kernel only when
`cache_len <= source_tile`; a multi-tile request deliberately falls back to
the established direct attention kernel. This is a correctness containment
fix, not a performance promotion.

The frozen criterion was copied unchanged from the preceding NO-GO gate. Its
SHA-256 is
`645df099030dcf3beca1289e0cc848f0f9c53c1725866896e06848631d962978`:

| Requirement | Frozen threshold |
| --- | ---: |
| Greedy generated tokens | exact match |
| Finite values | all values |
| max absolute difference | <= 2e-5 |
| relative L2 difference | <= 1e-5 |
| cosine similarity | >= 0.999999 |

`summary.json` records 24 direct-versus-candidate vector pairs for each tile;
both routes have exact tokens, all finite values, and maximum vector `max_abs`
of `0.0` across all 48 comparisons. The evaluator exited zero without
relaxing the criterion.

## Root-cause diagnosis

The original hypothesis that g0001 writes an incorrect KV cache state is
refuted for the tested first-decode transition. The unfixed direct and tile128
routes were run on the 128-token real prompt and their written logical KV
prefixes were captured after g0001 (`cache_len=129`). Across all 40 layers,
both K and V contain 132,096 F32 elements per layer and are bit-identical:

| KV diagnostic | Result |
| --- | --- |
| all values finite | true |
| all K/V prefixes bit-identical | true |
| worst max abs / relative L2 | `0.0` / `0.0` |
| first difference | none |

See `diagnostics/unfixed-kv/summary.json` and the hash-checked F32LE captures.
Thus the first large g0002 difference cannot be caused by a divergent KV
prefix written after g0001. It occurs while the two routes consume the same
state.

The explicit F32 API sweep localizes the trigger to `split_count > 1`, rather
than a tail being required:

| cache length | tile | split count | max abs vs direct |
| ---: | ---: | ---: | ---: |
| 128 | 128 | 1 | 0.0 |
| 129 | 128 | 2 | 1.4901e-8 |
| 256 | 128 | 2 | 1.63913e-7 |
| 257 | 256 | 2 | 7.451e-9 |
| 512 | 128 | 4 | 1.11759e-7 |
| 512 | 256 | 2 | 9.6858e-8 |

In particular, C=256 and C=512 are both exact source-tile multiples for the
listed routes, so a source-tail handler is not necessary for the mismatch.
They are also page-size multiples (the page is 16 tokens), so a tail/page
interaction is not necessary for it either. An independent fault-injection
proof excluding every possible page-indexing defect was not performed; that
broader claim remains **未確認**.

The split kernel computes each tile's online-softmax `(max, denominator,
weighted value)` state independently, then rescales and merges those partial
states. The direct kernel carries one online-softmax state through all source
tokens. These are mathematically equivalent but have different F32 operation
associations once more than one tile is present. The API sweep measures the
resulting small difference. The SQ8 serving stack applies four activation
quantizations per layer (160 over 40 layers), and the preceding full-model
gate showed that an initially small g0001 vector difference can become a large
g0002 difference. Together with the bit-identical g0001 KV prefix, this
establishes a **multi-tile partial-merge numerical-semantics defect relative to
the direct SQ8 serving contract**, not a g0001 KV write-position or range
defect.

The exact first activation-quantizer threshold crossing was not separately
captured, so that micro-mechanism is **未確認**. It is not needed for the
containment decision: the observed multi-tile result violates the frozen
full-model contract.

## Fix

Only the explicit source-tile experiment changes. In `PagedDecodeState`, a
configured tile uses the split body for a single tile and invokes the existing
direct paged-decode body whenever the cache needs multiple source tiles. The
normal environment-absent direct selection, legacy direct dispatch, and
runtime/kernel ABI are unchanged. The serving result marks an opt-in with
`direct-fallback-exact-state.v1` so a passing gate cannot be misread as a
passing multi-tile implementation.

The diagnostic support is test-only: it reads the written logical prefix in
logical block order, writes F32LE evidence after g0001, and compares SHA-256
validated layer captures. It is not an execution selector.

## Re-gate and performance

The re-gate uses the same real prompts and cache geometries: 127/128-token
prompts cover cache lengths 128--131, and 511/512-token prompts cover
512--515. Each request has three captured M=1 feedback steps with final hidden
state and logits compared to direct.

The performance run uses the canonical `raw-p0512` prompt and synchronized
generated indices 1--7 (cache lengths 513--519); model load, prefill, reset,
and oracle capture are excluded.

| route | mean M=1 ms | median M=1 ms | speedup vs direct | token exact |
| --- | ---: | ---: | ---: | --- |
| direct | 54.277224 | 50.515862 | 1.0000x | yes |
| tile128 | 55.596485 | 50.755068 | 0.9763x | yes |
| tile256 | 54.017278 | 50.316713 | 1.0048x | yes |

The earlier `1.2365x` tile128 result is therefore **not maintained**. All
measured cache lengths are above both source tiles, so the fixed candidates
take the direct fallback; the near-1x values are expected and are not a
performance claim. Telemetry also reported throttle states during this window,
so even these near-1x timing values remain conditional.

## Measurement window and operating record

One stop/isolated/restore window was used:

| Event | JST |
| --- | --- |
| window start | 2026-07-26 07:09:41 |
| preflight and service stop | 07:09:42 |
| measurements complete | 07:19:21 |
| initial-start restore succeeded | 07:19:21 |

Preflight verified GPU 2 as R9700 `gfx1201` at `0000:47:00.0`; V620 was not
selected. `llama-qwen35-udq4.service` was inactive/disabled and `gdm3` was
inactive before the stop. The final service state is `active/running`,
`Result=success`, and `NRestarts=0`; no reset-failed recovery was required.

There were 495 one-second R9700 telemetry samples. The sampled ranges were
8--326 W socket power, 3--3431 MHz gfx, 37--58 C edge, 38--76 C hotspot, and
34--62 C memory. AMD SMI displayed THROTTLED on 223 samples and UNTHROTTLED on
272. The raw v1.3 status fields and their header-derived interpretation are
preserved in `telemetry-summary.json`; the sustained physical cause is
**未確認** because the reason counters are unavailable and the status fields
were sampled separately. This state affects timing confidence, not the
geometry-specific numerical diagnosis.

No active-model manifest, campaign, authorization, `/opt/ullm` content,
systemd unit, or permanent GPU setting was changed. The artifact and package
identities, binary hashes, raw captures, service events, and telemetry are
retained in this directory.
