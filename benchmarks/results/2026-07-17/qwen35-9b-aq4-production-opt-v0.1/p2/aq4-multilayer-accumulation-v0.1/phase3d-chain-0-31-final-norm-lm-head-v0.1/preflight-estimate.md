# Phase 3d CPU-only full-chain preflight estimate

## Scope and safety boundary

- The run will use the existing 3-context Phase 1 hybrid fixture and the CPU-only `ullm-aq4-layer0-family-isolation` chain mode.
- It will measure decoder layers `0:31`, then final RMSNorm, then the LM-head projection at 34 fixed token rows (`0..31`, `220`, `41330`).  It will not materialize a vocabulary-sized logits tensor.
- GPU, production service, systemd, active manifest, the paused P3 harness, and every service-stop-window tool/evidence path are out of scope.

## Time estimate

The prior CPU-only `0:11` run completed 12 decoder layers in `157.13 s`.  Linear scaling gives:

`157.13 s * 32 / 12 = 419.01 s` (about 7 minutes)

Final RMSNorm is 9 x 4096 elements.  The sampled LM head decodes only 34 x 4096 BF16 weights once and executes 9 short dot-product batches; both are negligible relative to the 32 decoder layers.  Source shards first needed by later layers may add I/O, so the single run is given a conservative **20-minute timeout**.  It is not retried automatically on timeout or failure.

## Memory estimate

The prior `0:11` measurement had maximum RSS `332008 KiB` and `Swaps: 0`.  The chain retains only the current sequence and layer-local state; it does not retain all 32 layer outputs.  The extension adds at most:

- final-norm vector: `4096 * 4 = 16 KiB` per current timestep;
- LM-head sampled rows: `34 * 4096 * 4 = 557056 bytes` in f32;
- sampled logits: `34 * 4 = 136 bytes` per current timestep.

Therefore expected RSS remains near the prior result and is conservatively budgeted below **512 MiB**, not proportional to the 32-layer range.  Before the run the host reported about `71 GiB` available memory.  The system already had swap in use, so `/usr/bin/time -v` will record this process's `Swaps` count and no automatic retry will hide an unexpected result.

## Decision

One contiguous `0:31` measurement is safe to attempt.  Splitting at layer 16 is deliberately not used: the current fixture contains embedding residuals, and a run starting after layer 0 would not receive mathematically valid predecessor hidden states.  A process-level full chain preserves the required decoder state and bounded-memory behavior.
