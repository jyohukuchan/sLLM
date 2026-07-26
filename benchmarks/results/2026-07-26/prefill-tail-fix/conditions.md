# Measurement conditions

## Matching AY

The changed uLLM path was rerun using the same engine-loop contract as
[`../r9700-prefill-comparison/conditions.md`](../r9700-prefill-comparison/conditions.md):

| property | tail-fix condition |
| --- | --- |
| device | R9700 only: AMD SMI GPU 2, `0000:47:00.0`, `gfx1201` |
| visibility | `HIP_VISIBLE_DEVICES=1`, `ROCR_VISIBLE_DEVICES` unset |
| model | Qwen3-14B SQ8_0 artifact `2243acf1…c98b9147`; package manifest `c2133dfe…b63d1a0eb` |
| prefill mode | `m128-chunk128`, F32 K/V, single sequence |
| prompt lengths | 128, 512, 1024, 2048, 4095 |
| timing | one same-length unprofiled warm-up, then five timed repeats |
| timer | driver `Instant` around synchronized prefill advances only |
| excluded | model load, warm-up, request start, finish/reset, profiler duration |
| guards | HIP-only RMSNorm, RoPE, causal attention, add, SiLU-mul, paged KV write/decode attention, cached-prefix Flash2, BF16 matvec, and BF16 row guards |
| thermal gate | before each condition: edge ≤40°C, hotspot ≤42°C, socket power ≤35W; five-second polling |
| telemetry | `amd-smi metric --gpu 2 --json` at about one-second cadence during each process |

The candidate driver was built from an isolated clean base plus only the
tail-fix production changes.  This prevents concurrent changes to attention
sources from contaminating the result; see [`source-isolation.md`](source-isolation.md).

## Thermal record

Every start gate passed.  The columns below are the accepted gate sample;
the full min/max clock, temperature, power, and throttle observations are in
each condition’s `amd-smi-metric.jsonl`.

| prompt | edge °C | hotspot °C | memory °C | socket W | gfx MHz | throttle at gate |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 128 | 40 | 42 | 40 | 16 | 31 | UNTHROTTLED |
| 512 | 40 | 41 | 40 | 15 | 91 | UNTHROTTLED |
| 1024 | 40 | 41 | 38 | 8 | 7 | THROTTLED |
| 2048 | 40 | 42 | 40 | 17 | 1141 | UNTHROTTLED |
| 4095 | 40 | 41 | 40 | 13 | 48 | UNTHROTTLED |

As in AY, the gate applies before model load and warm-up.  The timed portions
heat the card and include both `THROTTLED` and `UNTHROTTLED` SMI statuses;
therefore this is a same-protocol comparison, not a claim of
temperature-normalized timed starts.  Per-condition maximums are retained in
the raw telemetry (4095 reaches 74°C edge / 98°C hotspot / 474 W in this
coarse one-second sampling).

An unrelated Gemma4 GPU trace was observed during the **2048 cooldown only**.
The candidate driver was not running then; it had exited before the accepted
gate and the 2048 candidate launch.  No foreign process was observed during
the 2048 or 4095 timed driver intervals.  This observation is recorded to
avoid implying a perfectly exclusive whole-window GPU state.

## llama.cpp comparison row

The llama.cpp Q8_0/F32-KV values in the README are carried from AY rather
than rerun here.  They are an unchanged comparator under AY’s documented
`-ngl 999`, `-fa on`, `-t 1`, `-b N`, `-ub 128`, F32-KV, five-repeat prompt
test.  This tail-fix window ran the changed uLLM side only and avoided opening
another service/GPU window for an unchanged executable.
