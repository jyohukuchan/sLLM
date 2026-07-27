# Gemma4 M=8 LDS BF16 matmul experiment (rejected)

This is an append-only negative-evidence record for the Gemma-only HIPRTC
batched BF16 matmul experiment.

- Candidate: one 256-thread CTA per output row and eight M rows.  Each CTA
  stages a 256-element BF16 weight strip in LDS once; its eight wave32s
  consume that strip for eight input rows.
- Clean R9700/gfx1201 rebuild, layer-major path opt-in: N=128 prefill was
  16.673954 tok/s (59.973777 ms/token).  The retained scalar-batch prototype
  was 16.662306 tok/s; the promoted PLE-only route was 18.683 tok/s.
- The candidate passed the real-activation differential and causal-prefix
  validation.  The validation generated the known cached continuations
  `[9079, 236761, 108, 818]` and `[528, 496, 1902, 1298]`.
- The N=128 benchmark retained 4,480 attention launches and measured decode
  at 8.162753 tok/s, versus the prior 8.161170 tok/s.  Decode does not invoke
  this M>1 kernel.
- N=512/N=2048 were intentionally not completed after N=128 failed the
  18.683 tok/s promotion gate; the in-flight N=512 run was terminated before
  it emitted an output, and the lock was released.  The default remains the
  promoted PLE-only path.
- Qwen3.5 AQ4_0 worktree probes before and after are byte-identical:
  `30865287e7525f4b24449ec24be3aa7619bfbbbbf48522cf2f67f9e58379b588`, top-1
  `220 / 8.529029846191406`.

Raw artifacts are in `raw/`.
