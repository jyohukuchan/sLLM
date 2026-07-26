# Measurement conditions and accounting

## Scope

This evidence uses only the R9700: AMD SMI GPU 2, PCI `0000:47:00.0`,
`gfx1201`, with `HIP_VISIBLE_DEVICES=1`. No workload was run on the V620.
The model is Qwen3-14B `SQ8_0`, F32 K/V cache, one sequence and no request
concurrency. Prompt sizes are 128, 512, 1024, 2048, and 4095; 4095 reserves
one generated token inside the 4096-token context.

The full-model candidate/control timing follows
`../r9700-prefill-comparison/conditions.md` and `accounting.md`:

- fixed M=128 chunks, one same-length unprofiled warm-up, then five timed repeats;
- the driver uses a synchronized `Instant` interval, excluding load, warm-up,
  request construction, and finish/reset;
- each condition waits for edge <= 40 C, hotspot <= 42 C, and socket power <=
  35 W before beginning; AMD SMI metrics are retained during the run;
- generic control and serial GQA use the same candidate executable. The control
  alone sets `ULLM_DISABLE_SQ8_0_FLASH2_GQA_GROUPED=1`.

The llama.cpp Q8_0/F32-KV throughput values in the result table are the
existing AY/BK comparison reference under that same accounting. The Phase 0
llama.cpp profiler capture uses the F32-KV configuration and is a separate
composition measurement; it is not used as a throughput substitute.

## Profiler boundary

`rocprof` reports per-kernel duration, not prefill tok/s. uLLM aggregates all
kernel rows in the driver's selected region. llama.cpp selects the terminal
interval bounded by its last two long `hipStreamSynchronize` calls. Kernel
duration sums are used only for within-trace composition and launch geometry.
They must not be divided into prompt tokens or compared across the two
different trace boundaries as an engine throughput number.

## Tail and numerical scope

The BK cursor-rewind tail fix remains in place. The serial GQA implementation
does not edit `crates/ullm-engine/src/sq8_serving_runtime.rs`. The 4095-token
oracle record verifies expected cache lengths on every layer and 32 prefill
advances, including the final 127-token request remainder.

For every target length, the generic and candidate oracle captures include
final hidden state, logits, top-1 token, generated token, non-finite counts,
and cache execution-unit metadata. The comparison records metrics for review;
it deliberately does not turn a scalar numerical value into a promotion gate.

## Service isolation

Before each GPU window, the command records `fuser` for `/run/ullm/r9700.lock`,
the required process scan, and service state. It never steals a held lock.
The wrapper stops `ullm-openai.service` once, waits for the lock to become
free, obtains `flock`, and restores the service from its EXIT trap. The
disabled `llama-qwen35-udq4.service` is checked inactive/disabled and never
started. No manifest, systemd unit, or `/opt/ullm` file is written.
