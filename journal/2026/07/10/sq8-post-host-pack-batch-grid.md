# SQ8 Post Host-Pack Batch Grid

- Re-ran the Qwen3-14B-FP8 SQ8_0 b2/b4/b8 normalized comparison rows after first-layer residual host staging was reduced.
- New result directory:
  - `benchmarks/results/2026-07-10/sq8-qwen3-14b-post-host-pack-refresh/`
- Commands were executed through `tools/run-external-benchmark.py --parse ullm-model-loop-throughput` with `ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_BATCH_KERNEL=1`.
- All rows used current commit `7af8c3a`, `/tmp/ullm-qwen3-14b-fp8-bf16-thin.ullm.d`, and `/tmp/ullm-qwen3-14b-fp8-full-sq8-artifact`.
- Key results:
  - b2: `sq_fp8_batch_matvec_count=6720/6720`, host staging `0/24`, write bytes `983040`, decode `16.596278` tok/s.
  - b4: `sq_fp8_batch_matvec_count=6720/6720`, host staging `0/24`, write bytes `1966080`, decode `16.778691` tok/s.
  - b8: `sq_fp8_batch_matvec_count=6720/6720`, host staging `0/24`, write bytes `3932160`, decode `16.689519` tok/s.
- Verified the combined uLLM/vLLM table with:
  - `python3 tools/summarize-sq8-vllm-batch-grid.py benchmarks/results/2026-07-10/sq8-qwen3-14b-post-host-pack-refresh/results.jsonl benchmarks/results/2026-07-09/sq8-vllm-fp8-comparison/results.jsonl --workload-prefix pp16-tg8 --requests 2,4,8 --require-normalized-throughput-comparison --require-ullm-sq-batch-coverage --require-ullm-sq-kernel-families --show-sq-details`
- Added `--max-ullm-sq-host-staging-write-count 24` to the summary gate so this reduced write-count shape is regression-checkable.
