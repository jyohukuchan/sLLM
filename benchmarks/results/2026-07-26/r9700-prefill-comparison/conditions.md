# Measurement conditions (fixed before execution)

## Scope

This is a single-stream (`batch=1`, no concurrent requests) **prompt
processing/prefill** comparison of Qwen3-14B approximately-8-bit weights on
the R9700 only (AMD SMI GPU 2, `0000:47:00.0`, `gfx1201`).  It is not a
quality comparison: uLLM uses the supplied SQ8_0 artifact while llama.cpp uses
the supplied GGUF Q8_0 file.

The target prompt lengths are `128, 512, 1024, 2048, 4095`.  `4095` is the
operational 4096-point: the uLLM serving request necessarily reserves one
generated token so `4096 + 1` cannot fit the 4096-token context.  Every uLLM
request therefore uses `max_new_tokens=1`; the timed interval ends when that
token has been produced and does not include a subsequent decode step.  The
matching llama.cpp test is `-p N -n 0` and evaluates the prompt only.

Each condition has one unprofiled, same-length warm-up and five timed repeats.
The uLLM timer excludes model load, warm-up, request construction, and
finish/reset.  llama-bench loads and warms before its repetitions; source
inspection shows its `t_start` follows warm-up/model setup and its prompt test
ends with `llama_synchronize`.

## Common conditions

| property | selected condition |
| --- | --- |
| device isolation | uLLM: `HIP_VISIBLE_DEVICES=1`; llama.cpp: `env -u ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=1`; both validate one visible gfx1201 device |
| device | R9700 AMD SMI index 2 / PCI `0000:47:00.0`; no V620 workload is permitted |
| base model | Qwen/Qwen3-14B; GGUF file SHA-256 is captured in `environment/gguf-sha256.txt` |
| stream / parallelism | one sequence, no request concurrency |
| timed repetitions | five, after one same-length warm-up |
| projection microbatch | uLLM fixed M=128 chunks; llama.cpp `-b N -ub 128`, making each prompt one `llama_decode` call and internal 128-token ubatches |
| GPU residency | uLLM artifact is uploaded by its session loader; llama.cpp uses `-ngl 999`, validated in its result JSON/log |
| flash attention | uLLM `m128-chunk128` F32 cached-prefix Flash2 guard; llama.cpp `-fa on`, validated in output |
| host CPU | llama.cpp `-t 1`; no compilation during the service window |
| thermal gate | before every condition: hotspot <= 42 C, edge <= 40 C, socket power <= 35 W; five-second polling until gate passes |
| thermal record | `amd-smi metric --gpu 2 --json` at approximately one second intervals during every load/warm-up/timed process; raw JSONL retained |

## KV variants and deliberate differences

uLLM uses F32 K and V (source excerpt:
`environment/ullm-kv-layout-source.rs`).  llama.cpp is measured twice at every prompt length:
F32 K+V and F16 K+V (`--cache-type-k/--cache-type-v`).  The F16 row is a
practical llama.cpp variant, not byte-identical KV storage to uLLM.  The
normalized logical traffic metric in `accounting.md` intentionally retains the
same F32-equivalent causal KV denominator for all three rows; a separate
format-aware lower-bound byte column is retained in `summary.json`.

The models have the same declared base architecture but not identical encoded
weights or output-head dtype: uLLM has FP8 SQ8_0 projections with BF16 scales
and BF16 LM head, whereas the GGUF Q8_0 matrices (including its output head)
are Q8_0.  No claim of bit-identical logits or quality is made.

The timing APIs are necessarily different: uLLM uses an internal serving
session driver (`greedy_ignore_eos_for_testing`) and llama.cpp uses
`llama-bench`.  Both exclude model load and the same-length warm-up from their
reported five repetitions, and neither result includes HTTP, tokenization, or
server scheduling.  This is an engine-loop prefill comparison, not an
end-to-end API-latency comparison.  The distinct session/reset implementations
remain an unmatched harness difference.

The local llama.cpp tree is at commit `68a5592` plus a pre-existing three-line
`f32` cache-type parser mapping used for the F32 KV row.  The binary hash and
exact patch are retained in `environment/llama-build-identity.md`; it does not
alter a kernel, graph, model loader, cache layout, or timing code.  The F16
row uses the existing upstream parser path.

## Thermal outcome and limitation

The pre-process thermal gate passed for all fifteen conditions: edge was
38--40 C, hotspot 38--42 C, memory 36--40 C, and socket power 7--16 W at
the gate sample.  This is recorded per condition in raw/*/thermal-gate.json.

However, the timer begins after each implementation's same-length warm-up.
The one-second AMD SMI stream therefore shows that a strictly
temperature-matched **timed** start was not verified, especially for the long
uLLM runs.  For example, the nearest post-warm-up sample for uLLM N=2048 was
57 C edge / 78 C hotspot and for N=4095 was 69 C / 90 C, whereas the matching
llama.cpp F32 samples were 46 C / 62 C and 49 C / 66 C.  Those samples are
nearest-after-marker observations rather than hardware-synchronized readings;
at N=128 the entire timed run is shorter than a sensor interval.

Thus the gate materially improves on the earlier 73 C versus 36--39 C
pre-process mismatch, but it does **not** establish a temperature-normalized
long-prompt comparison.  The thermal envelope is retained rather than
retroactively claiming alignment; no additional service window was opened
solely to alter this condition.

## llama-bench timing verification

At llama.cpp commit `68a5592`, `tools/llama-bench/llama-bench.cpp` constructs
`-p` rows with `n_gen=0`; `test_prompt()` calls `llama_decode()` and then
`llama_synchronize()`.  The benchmark's `t_start` is set only after model load
and its prompt warm-up.  Thus model load, graph allocation/setup, and warm-up
are outside the five reported prompt timings.

`llama_batch_allocr::init()` marks only the last token as an output when
`llama_batch_get_one()` supplies no logits array.  With `-b N -ub 128`, this
means one output/LM-head evaluation at the final prompt token, matching the
uLLM one-token request boundary rather than one LM-head evaluation per 128-token
chunk.  The inspected source excerpts are saved under `environment/`.

## Service and safety protocol

Immediately before the isolated window the records must show
`llama-qwen35-udq4.service` `inactive` and `disabled`.  It is only inspected,
never started.  `ullm-openai.service` is stopped once for the entire window
and the wrapper attempts restoration from an EXIT trap; the final service
state must be recorded rather than inferred from the attempt.

Outcome addendum: the wrapper's `sudo -n` restoration attempt returned 1
after the sudo credential expired.  The service was observed active at
20:00:24, then stopped at 20:05:57 after an `unexpected worker stdout EOF`
record.  An approved explicit start at 20:06:13 restored it to active/running
at 20:06:14.  Further worker-EOF/restart events occurred during later normal
service operation.  The 20:17:27 EOF left the unit `start-limit-hit`; one
approved `reset-failed` plus explicit start at 20:19:34--20:19:35 restored it,
and the 20:20:10 audit was active/running.  This did not add an isolation
window: no second stop and no further benchmark process occurred.  No active
manifest, systemd unit, `/opt/ullm`, or V620 is modified.
