# Gemma4 definitive before/after measurement — EH

This directory records one R9700-only, exclusive measurement window. Both
uLLM endpoints are full-`cargo clean` release rebuilds, with build logs and
native-runtime artifact inventories retained. Each context uses five timed
prefills and five timed 128-token decode sequences, after an excluded warmup;
the report derives a median and min--max spread from the five per-run values.

The pre-residency worktree is `/tmp/ullm-gemma4-pre-residency-dn` at
`3a138a46`. Its uncommitted `ullm-gemma4-resident` diff adds the neutral
benchmark CLI parameters needed to issue exactly the same fixed repeated
token-id-2 workload as current HEAD; it does not alter executor behavior.
Current is `b9c899c6` with the promoted defaults.

llama.cpp is rerun serially in the same window with `gemma-4-E2B-BF16.gguf`,
all layers offloaded, F32 K/V, Flash Attention off, and five repetitions. Its
prefill row is `-p N -n 0`; its matched 128-token generation row is
`-p 0 -n 128`, preserving the established comparable contract.

`ullm-openai` is stopped before lock acquisition and restarted only after the
user benchmark process exits and releases the lock. Telemetry records R9700
GPU-2 hotspot (junction) samples every five seconds during every timed run.
No profiler is used and no physical HBM counter is claimed.
