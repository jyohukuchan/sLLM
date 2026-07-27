# CO: SQ8_0 tile-128 quality and R9700 hardware measurement

## 前回の要点

- CN had already produced the full-model SQ8_0 numeric comparison and the
  successful Qwen3.5-35B-A3B MoE admission. Repeating either would consume the
  exclusive device window without adding the missing decision evidence.
- The remaining blockers were the candidate text suite (the capture process had
  previously lacked `ULLM_SERVED_MODEL_MANIFEST`) and the first R9700 runtime
  microbenchmark.

## 今回の変更点

- Before stopping production, the actual lightweight harness path was run
  read-only against the active AQ4_0 gateway: two selected cases from the
  validated eight-case CF suite, each captured twice and compared into Markdown.
  It passed with blocking 0, comparison pass true, no service operations, and
  unchanged active SHA `a654d92fe8142fcc0904fe187c96b84c95e0dd18acac61ef25d0cfa6429a08cd`.
- One exclusive R9700 window then ran the missing tile-128 candidate suite.
  The fixed manifest environment reached readiness and generated all eight CF
  cases. The comparison has no automatic blocking findings. In particular,
  `javascript_debug_extended` correctly says `NaN` is falsy, `Infinity` is
  truthy, the original Boolean filter therefore keeps three values, and the
  `isFinite` fix returns two. This is a quality pass under the lightweight
  policy; exact output match is 0.000 only as a diagnostic. SQ8_0 was not
  activated.
- The same lock then built, ISA-audited, and measured gfx1201. STREAM copy and
  triad reached 584.167 and 574.355 GB/s; real-shape Qwen3-14B GEMM reached
  BF16 15.205 and FP8 23.535 TFLOPS. All performance values use HIP events,
  5 warmups, median of 11 samples, and 10 launches/sample.
- The first hardware wrapper had two evidence-output defects: it used unsupported
  `amd-smi metric -j`, so group telemetry files contain that error, and the
  validation line went to stdout instead of the requested JSONL. It nevertheless
  completed validation before the subsequent groups (a validation mismatch would
  have aborted the wrapper). Both output paths are corrected for a MI300X fill-in
  run. The enclosing window telemetry records start edge 45 C / GFX 234 MHz /
  13 W and post-group edge 48 C / GFX 430 MHz / 13 W; no value was invented for
  the missing per-group telemetry.
- The runner's immediate container restore probe again returned
  `container_transport` despite service state active. A subsequent direct
  OpenWebUI bridge completion returned HTTP 200 / `restored`; active SHA stayed
  unchanged and `NRestarts=0`.

## 次の行動

- SQ8_0 tile-128 text quality no longer blocks its speed evidence, but this task
  makes no production switch.
- On MI300X, run the now-corrected wrapper once to fill the pending comparison
  column, including persisted CPU-validation values and valid per-group telemetry.
