# R9700 prefill comparison: uLLM SQ8_0 vs llama.cpp Q8_0

Date: 2026-07-26.  Device: AMD SMI GPU 2, PCI 0000:47:00.0, gfx1201
R9700 only.  Every completed row is one stream, one same-length unprofiled
warm-up, and five timed repetitions.

This is a speed-positioning comparison, not a quality claim.  The inputs share
the Qwen/Qwen3-14B base model, but uLLM uses the supplied SQ8_0 artifact and
llama.cpp uses GGUF Q8_0.  The GGUF identity is recorded in environment/.

## Result

Each cell is tok/s; logical GB/s; lower-bound TFLOP/s; logical 640-GB/s roof
ratio.  The roof ratio is not a physical-HBM efficiency; see Accounting below.

| prompt tokens | uLLM SQ8_0 / F32 KV | llama.cpp Q8_0 / F32 KV | llama.cpp Q8_0 / F16 KV |
| ---: | --- | --- | --- |
| 128 | 851.659; 188.550; 22.560; 29.461% | 1165.756; 258.088; 30.880; 40.326% | 1189.076; 263.251; 31.498; 41.133% |
| 512 | 513.676; 270.629; 13.683; 42.286% | 1195.722; 629.963; 31.851; 98.432% | 1187.016; 625.377; 31.619; 97.715% |
| 1024 | 335.996; 317.435; 9.020; 49.599% | 1145.351; 1082.081; 30.747; 169.075% | 1174.539; 1109.656; 31.531; 173.384% |
| 2048 | 188.425; 335.936; 5.137; 52.490% | 1058.379; 1886.942; 28.856; 294.835% | 1127.775; 2010.665; 30.748; 314.166% |
| 4095 | 71.576; 247.610; 2.011; 38.689% | 1008.683; 3489.444; 28.346; 545.226% | 1054.871; 3649.230; 29.644; 570.192% |

The prompt 4095 row is the operational 4096-point: uLLM reserves one output
token, so 4095 prompt plus one output equals the 4096-token context limit.
The uLLM driver starts that one-token request, but its timed loop calls
`advance_synchronized` only while the session is in `Prefilling`; the final
prompt forward transitions it to `Finishing`, and no `Decoding` advance is
timed.  llama.cpp uses prompt-only p=N, n=0.  Thus an exactly-4096 prompt
with a uLLM output token was not measured, rather than silently exceeding its
advertised context.

## Direct comparison

uLLM does not win any measured prompt length.

| prompt tokens | llama F32 / uLLM | llama F16 / uLLM |
| ---: | ---: | ---: |
| 128 | 1.369x | 1.396x |
| 512 | 2.328x | 2.311x |
| 1024 | 3.409x | 3.496x |
| 2048 | 5.617x | 5.985x |
| 4095 | 14.092x | 14.738x |

At N=1024, uLLM measured 335.996 tok/s, close to the earlier unprofiled
337.132 tok/s, which checks that the current synchronized harness is
consistent with the earlier uLLM prefill observation.  It nevertheless loses
to both llama.cpp KV choices at that point.

The 4095 result has an important execution-path distinction.  uLLM fixed-M128
prefill executes 31 full 128-token units and then falls back to 127 M=1 units
for the remainder, as confirmed by 158 advance calls in every repetition.
llama.cpp records 32 internal 128-token ubatches.  This is an actual uLLM
serving-path behavior at this boundary, not a corrected-away artifact; it is
also why the 4095 ratio must not be generalized to an exactly divisible M128
prompt length.

## Accounting and bottleneck interpretation

The common byte numerator and lower-bound FLOP numerator are defined in
[accounting.md](accounting.md).  It preserves the prior SQ8 logical policy:
280 projection payloads and BF16 scales, one BF16 LM head, and
F32-equivalent Q-head-expanded causal GQA KV traffic.  The exact N=1024
causal KV term is 859,832,320,000 B, matching the earlier attention study.

The common logical KV share is 47.9% at N=128 and 79.8%, 88.9%, 94.2%, and
97.0% at N=512 through 4095.  Thus the accounting identifies uLLM long-prompt
work as causal-attention/KV-traffic dominated; N=128 is mixed with projection
work.  It does not prove a physical bandwidth bottleneck.  No HBM/TCC counter
was captured, and logical roof ratios above 100% for llama.cpp demonstrate
that GQA reuse/fusion makes this numerator unsuitable as a physical HBM
efficiency.  Physical HBM efficiency and a strict global memory-versus-compute
bottleneck classification are therefore unconfirmed.

The achieved lower-bound FLOP indicator reinforces the observed contrast:
llama.cpp sustains roughly 28--32 TFLOP/s through the sweep, while uLLM falls
from 22.560 TFLOP/s at N=128 to 2.011 TFLOP/s at N=4095.  This is an achieved
work figure, not peak-FLOPS efficiency; norms, softmax, quantization,
activations, and other non-matmul work are excluded.

## Timing method and matching

- uLLM used a standalone measurement-only driver built against clean HEAD
  0216b131, with SQ8 fixed-M128 cached-prefix prefill.  The exact driver
  source, Cargo manifest, source/binary SHA-256s, and clean-tree identity are
  retained in environment/.  Its timed loop is advance_synchronized only and
  excludes model load, request start, same-length warm-up, and finish/reset.
- llama.cpp used the local gfx1201 build at 68a5592, with p=N, n=0, r=5,
  batch=N, ubatch=128, all GPU layers requested, flash attention on, and
  one CPU thread.  Source excerpts in environment/ show that model load and
  warm-up occur before t_start and that test_prompt synchronizes.
  Its F32-KV parser has the pre-existing minimal mapping recorded in
  [environment/llama-build-identity.md](environment/llama-build-identity.md);
  it does not change the timing or kernel path.
- Passing batch=N lets llama-bench submit the prompt once while its 128-token
  internal ubatch remains matched to uLLM M128 for full chunks.  The last-token
  logits selection means one LM head per prompt, not one per ubatch.
- uLLM F32 KV is the primary comparison.  llama.cpp F16 KV is intentionally a
  storage-different practical variant.  Full conditions and all unmatched
  details are in [conditions.md](conditions.md).

## Thermal record

Before every process, the thermal gate passed at edge 38--40 C, hotspot
38--42 C, memory 36--40 C, and socket 7--16 W.  Raw samples are in
raw/cooldown/ and [thermal-history.csv](thermal-history.csv); the condensed
temperature, clock, power, and sampled throttle-status rows are in
[thermal-summary.md](thermal-summary.md).

The intended timed-start thermal alignment was not fully achieved because the
same-length warm-up occurs after that gate.  For long uLLM conditions the
nearest post-warm-up telemetry sample is materially warmer than llama.cpp:
at N=2048, 57/78 C edge/hotspot versus llama F32 46/62 C; at N=4095,
69/90 C versus 49/66 C.  These are one-second nearest samples, not
hardware-synchronized timestamps, and short N=128 timings are shorter than
the sensor interval.  Therefore this result is not a temperature-normalized
long-prompt causal comparison.  The observed thermal difference is retained
rather than hidden or corrected by assumption.

## Isolation and service lifecycle

There was one intentional isolation window: stop at
2026-07-26T19:12:23+09:00, with all 15 raw conditions complete at
19:59:57+09:00.  The wrapper's final `sudo -n systemctl start` returned 1
because its credential timestamp had expired.  systemd subsequently showed
the service active at 20:00:24, but the gateway then stopped at 20:05:57 after
recording `unexpected worker stdout EOF`.  An approved explicit start was
issued at 20:06:13.  During later normal service operation, the journal showed
further worker-EOF/restart events at 20:08:11, 20:09:06, 20:11:56, 20:12:42,
and 20:17:27; their cause is unconfirmed.  The final EOF left the unit
`start-limit-hit`.  An approved `reset-failed` plus one explicit start at
20:19:34--20:19:35 restored it; the 20:20:10 audit observed
active/running/enabled, MainPID 1480646, NRestarts=0.  No second stop or
measurement window was opened.  `llama-qwen35-udq4.service` remained
inactive/disabled and was never started.

The first failed start attempt is recorded verbatim in service/restore.txt.
The later worker-EOF observation and final restoration are retained in
[service/final-recovery.md](service/final-recovery.md), without asserting an
unconfirmed initiator for the 20:00:24 or 20:09:50 starts.  No active
manifest, systemd unit content, or file under /opt/ullm was modified.

## Evidence layout

- raw/: command text, stdout/stderr, timestamped stream events, per-condition
  thermal gate, and raw AMD SMI JSONL.
- environment/: build/artifact identity and inspected source excerpts.
- service/: the one stop window and final-state record.
- [commands.md](commands.md): credential-free model commands.
- [summary.json](summary.json), [comparison.csv](comparison.csv), and
  [r9700-prefill-comparison.jsonl](r9700-prefill-comparison.jsonl): normalized
  machine-readable results.
