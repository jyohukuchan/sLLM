# SQ8_0 resident stack diagnostic

## 前回の要点

- AQ4由来stack real-batch runnerへSQ8_0 overlayを載せると、scheduler上はreal-batchでもSQ8_0 projectionは `materialized_f32_fallback` になっていた。
- layer3 component batch smokeでは `sq_fp8_batch_matvec_count=14/14` を確認済みだが、これはfull-package stack rowではない。

## 今回の変更点

- 実装用subagent Archimedes (`gpt-5.3-codex-spark`) が `sq-fp8-package-self-attn-stack-batch-smoke` を追加した。
- 新コマンドは既存のmixed request-state resident pathを使い、SQ8_0 sidecarを `PackageAq4ResidentMatvec::load_with_sq_overlay` 側へ流す。
- `package_token_ids_mixed_request_state_smoke_impl_with_sq_overlay` のstdoutへ `sq_fp8_expected_all_batch_matvec_count` を追加した。
- parser testに、resident stack診断が `grouped` として保存され、`sq_fp8_batch_matvec_count=0/21` を保持するケースを追加した。
- layer3 Qwen3.5 SQ8_0 artifactで実機smokeを実行し、`benchmarks/results/2026-07-09/sq8-stack-resident-diagnostic/results.jsonl` に保存した。

## 実測結果

- `status=ok`
- `sq_execution_mode=direct_fp8_dequant_matvec`
- `sq_projection_boundary=single+triple`
- `sq_projection_implementation_ids=single=sq8_0_matvec_r9700_direct,triple=sq8_0_matvec_triple_r9700_direct`
- `batching.mode=grouped`
- `prefill_real_batch=false`
- `decode_real_batch=false`
- `sq_fp8_batch_matvec_count=0`
- `sq_fp8_expected_all_batch_matvec_count=21`

## 次の行動

- mixed request-state stack pathで、request groupingだけでなく実際の `matvec_batch` 境界を使う。
- 具体的には、q/k/v/o/gate/up/downのprojection呼び出しをrequest単位の `single/triple` から、同一layer同一stepのrequest batchをまとめた `matvec_batch` へ移す。
- その後、`sq-fp8-package-self-attn-stack-batch-smoke` が `sq_fp8_batch_matvec_count == sq_fp8_expected_all_batch_matvec_count` になるか確認する。
