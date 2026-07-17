# SQ8_0 M10 R9700 dispatch refresh

## 前回の要点

- M10比較にはQwen3-14B-FP8のconfig-aligned uLLM/vLLM行がある。
- ただし既存uLLM行はR9700 descriptor導入前に作られていたため、
  `sq_projection_implementation_ids` は `*_rdna4_direct` のままだった。

## 今回の変更点

- 最新の `target/debug/ullm-engine` で、config-aligned smoke `pp16/tg8/b1` と
  representative `pp512/tg128/b1` を再測定した。
- 追加したsmoke case:
  `ullm-r9700-qwen3-14b-fp8-sq8-smoke-pp16-tg8-b1-rope128-theta1e6-r9700dispatch`
- smoke結果:
  - `status=ok`
  - `sq_execution_mode=direct_fp8_dequant_matvec`
  - `sq_projection_boundary=single+triple`
  - `sq_projection_implementation_ids=single=sq8_0_matvec_r9700_direct,triple=sq8_0_matvec_triple_r9700_direct`
  - `prefill_tokens_per_second=2.754645`
  - `decode_tokens_per_second=2.702039`
  - `end_to_end_total_tokens_per_second=0.32139`
  - `vram_consumed_bytes=13763952640`
- 追加したrepresentative case:
  `ullm-r9700-qwen3-14b-fp8-sq8-rep-pp512-tg128-b1-rope128-theta1e6-r9700dispatch`
- representative結果:
  - `status=ok`
  - `sq_execution_mode=direct_fp8_dequant_matvec`
  - `sq_projection_boundary=single+triple`
  - `sq_projection_implementation_ids=single=sq8_0_matvec_r9700_direct,triple=sq8_0_matvec_triple_r9700_direct`
  - `prefill_tokens_per_second=2.967403`
  - `decode_tokens_per_second=2.858786`
  - `end_to_end_total_tokens_per_second=2.267478`
  - `vram_consumed_bytes=14242406400`
- `tools/run-external-benchmark.py` 経由で、`ULLM_REQUIRE_HIP_SQ_FP8_MATVEC*` のenv記録も
  両行の `artifacts.command` に残した。
- `benchmarks/results/2026-07-09/sq8-vllm-fp8-comparison/summary.md` と
  `same-model-readiness.md` を更新した。

## 次の行動

- M10の最終比較には、real-batchまたはserver-style uLLM経路が必要。
