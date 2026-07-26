# SQ8_0 adaptive prefill chunk selection

## Scope

This result records the production-default adaptive policy for the Qwen3-14B
`SQ8_0` serving path on the R9700 (`gfx1201`). It uses only real prompt
tokens: fixed-width tail units retain the existing cursor-rewind and
real-token commit behavior. No padding, fabricated row, or attention-mask
path was introduced.

The successful R9700 window is [`attempt-2`](attempt-2/). It uses one
same-length warm-up request and five unprofiled, synchronized prefill samples
per prompt length; the throughput values below are the median of those five
samples, not profiler-range time. Every condition waited for edge temperature
<=45 C before starting.

The root-level first attempt is retained as a transparent failed harness
attempt: the old CLI rejected repeated prompt lengths before any SQ8_0 GPU
execution (`throughput/p128.stderr`). The CLI now permits intentional repeats
and gives them distinct deterministic request IDs; the retry was kept in a
fresh `attempt-2` directory.

## Selection rule

`Adaptive` is now the default. A fixed-width override remains available
through `ULLM_SQ8_PREFILL_CHUNK_TOKENS` (and the explicit serving CLI mode).
The selected width is fixed for the lifetime of one request:

| prompt tokens N | selected M | measured-table rationale |
| ---: | ---: | --- |
| 1..511 | 128 | M=128 is the short-prompt winner; M=256 did not beat M=128 at its nearest useful comparison. |
| 512..1023 | 512 | M=512 narrowly wins the N=512 column (565.624 vs 562.525 tok/s). |
| 1024..2047 | 1024 | M=1024 wins the N=1024 column (388.258 tok/s). |
| 2048..4096 | 2048 | M=2048 wins N=2048 and N=4095 (232.765 and 126.686 tok/s). |

M=256 is deliberately not selected: the measured result is lower than M=128
at the relevant comparison. M=4096 is admitted only as a lower-runtime shape
check; serving rejects it because a normal request reserves one decode
position, so N=4095 has no legal all-real 4096-row prefill unit. The N=4095
tail is therefore M=2048 followed by a rewound M=2048 unit that commits the
remaining 2047 real tokens.

## Prefill throughput

| N | selected M | adaptive median tok/s | five-sample aggregate tok/s | BY measured winner tok/s | comparison |
| ---: | ---: | ---: | ---: | ---: | --- |
| 128 | 128 | **887.490** | 886.246 | 883.091 | maintains short-prompt result |
| 512 | 512 | **566.458** | 566.358 | 565.624 | matches measured winner |
| 1024 | 1024 | **388.173** | 386.606 | 388.258 | matches measured winner |
| 2048 | 2048 | **232.671** | 233.236 | 232.765 | matches measured winner |
| 4095 | 2048 | **126.791** | 126.454 | 126.686 | matches measured winner |

The raw synchronized durations and selected mode are in
[`attempt-2/throughput`](attempt-2/throughput/). In particular, N=128 is
887.490 tok/s median, so the M=2048 short-prompt fallback regression is not
present. At N=4095 the llama.cpp reference of 1008.683 tok/s is 7.955x this
result, reduced from the M=128 control's 9.610x gap; adaptive is 1.208x the
M=128 control at that length.

## Correctness and generation review

The direct fixed-M=128 versus adaptive oracle comparison is in
[`attempt-2/numerical/adaptive-vs-m128.json`](attempt-2/numerical/adaptive-vs-m128.json).
For N=128/512/1024/2048/4095, final hidden state and logits are F32
byte-identical (`max_abs=0`, no non-finite values), and the greedy token is
identical. The recorded adaptive execution widths are respectively 128, 512,
1024, 2048, and `2048 + tail 2047`.

The N=4000 long-prefix run generated the same 83 token IDs and 467 decoded
characters for fixed M=128 and adaptive M=2048. The text begins “Processing a
long prefill in fewer real-token chunks can reduce repeated work...” and ends
with a rollback-record verification step; raw results are
[`long-m128_chunk128.json`](attempt-2/generation/long-m128_chunk128.json)
and [`long-adaptive.json`](attempt-2/generation/long-adaptive.json).

The 10-case Japanese/English/code/summary/multiturn suite has 10/10 equal
token-ID sequences and decoded texts, with no empty candidate. The decoded
side-by-side review is
[`comparison.md`](attempt-2/generation/decoded-comparison/comparison.md) and
the machine-readable summary is
[`comparison.json`](attempt-2/generation/decoded-comparison/comparison.json).
That reused decoder has generic display headings (“CK baseline” and
“Handwritten WMMA candidate”); here its actual input directories are
`generation/m128_chunk128` and `generation/adaptive`, respectively. Manual
review found no adaptive regression or garbling. One inherited baseline
response in `javascript_debug` incorrectly says `Boolean(Infinity)` is false;
the adaptive text is exactly the same, so this is a model-content issue, not
a change-induced divergence.

Fresh paged decode at N=1024 measured **27.592484 tok/s**
([`attempt-2/decode/p1024.json`](attempt-2/decode/p1024.json)), preserving
the 27.552769 tok/s reference measurement.

## VRAM behavior

Adaptive loading starts with M=128 allocations. At a Ready/reset baseline it
atomically replaces only the M-dependent stack workspace, resident hidden
state, and prompt chunk buffer for the selected M; weights, K/V cache, and
decode state remain resident. Thus a short request does not reserve the M=2048
prefill buffers at startup, and a subsequent short request reconfigures back
to M=128. Replacement allocation occurs before dropping the old buffers, so
a width transition briefly holds old and new width-dependent buffers; it does
not duplicate model weights.

The directly comparable allocation ledger remains BY's measured data: M=128
16.525 GiB, M=512 16.718 GiB, M=1024 16.975 GiB, and M=2048 17.489 GiB.
M=2048 is therefore +0.964 GiB over M=128, with the recorded AQ4_0
co-resident headroom 7.453 GiB. A live `amd-smi process` observation of an
adaptive M=128 suite request is retained at
[`attempt-2/vrams/adaptive-m128-ja-multiturn.json`](attempt-2/vrams/adaptive-m128-ja-multiturn.json);
its 13,449,801,728-B process-accounting value is not treated as a replacement
for the allocation ledger because it uses different accounting.

## Operations and promotion

There were two service-isolation attempts. The first was the pre-compute CLI
validation failure above; its `systemctl start` returned nonzero, but its
post-window service record was active with `NRestarts=0`. The successful
attempt-2 window ran from 06:47:42 to 07:38:15 JST, restored
`ullm-openai.service` successfully, and ends with `NRestarts=0` in
[`attempt-2/service/after-restore-ullm-openai.txt`](attempt-2/service/after-restore-ullm-openai.txt).
`llama-qwen35-udq4.service` remained inactive and disabled.

No served-model promotion was performed. Immediately after validation the
active manifest is the expected AQ4_0 manifest SHA-256
`3507102fd3015f47204a4f3256b818c58788eadb5573e5d5fe727a076cb1b3e7`, whose
worker is the root-owned, mode-0555 `/opt/ullm/.../ullm-aq4-worker` and whose
`execution.paged_decode_attention` remains `aq4_gqa_grouped_split` with
`split_tile: 128`. This task changes the SQ8_0 serving path; replacing that
active AQ4_0 model with SQ8_0 merely to publish the implementation would not
preserve the active model/execution configuration, so no promotion manifest
was created or applied.
