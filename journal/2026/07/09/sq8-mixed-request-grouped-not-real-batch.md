# SQ8_0 mixed request grouped reporting

## 前回の要点

- M10の同一モデル行はR9700 dispatch後の `*_r9700_direct` descriptorで更新済み。
- ただし最終的なvLLM serving比較には、uLLM側のreal-batchまたはserver-style pathがまだ必要。

## 今回の変更点

- `sq-fp8-token-ids-mixed-request-state-smoke` の複数request出力を、real-batchではなく
  grouped request executionとして表現するようにした。
- mixed-request-stateは同じtimestepのrequestをまとめるが、layer内ではrequestごとに
  stepを呼ぶため、現時点ではbatched projection kernelを使うreal-batchではない。
- 出力変更:
  - `batching_mode=grouped`
  - `prefill_real_batch=false`
  - `decode_real_batch=false`
  - `prefill_request_grouped=true/false`
  - `decode_request_grouped=true/false`
  - `prefill_grouped_request_parallelism`
  - `decode_grouped_request_parallelism`
- `tools/run-external-benchmark.py` はこれらの新項目を `batching` に保存する。

## 検証

- `cargo fmt --all --check`
- `cargo check -p ullm-engine`
- `python3 -m unittest tests.test_external_benchmark_batch_parser`
- `python3 -m py_compile tools/run-external-benchmark.py tests/test_external_benchmark_batch_parser.py`
- `cargo build -p ullm-engine`
- 実機smoke:
  `target/debug/ullm-engine sq-fp8-token-ids-mixed-request-state-smoke /tmp/ullm-qwen3-14b-fp8-bf16-thin.ullm.d /tmp/ullm-qwen3-14b-fp8-layer0-sq8-artifact 2 1048576 0 len:4x2 1 1 1024 128 1000000 0`
- 実機結果:
  - `batching_mode=grouped`
  - `prefill_real_batch=false`
  - `decode_real_batch=false`
  - `prefill_request_grouped=true`
  - `decode_request_grouped=true`
  - `prefill_grouped_request_parallelism=2`
  - `decode_grouped_request_parallelism=2`
  - `sq_fp8_batch_matvec_count=0`
  - `sq_projection_implementation_ids=single=sq8_0_matvec_r9700_direct,triple=sq8_0_matvec_triple_r9700_direct`

## 次の行動

- 本物のreal-batch化には、mixed-request-state layer path側でbatched projection kernelを使う実装が必要。
- M10最終比較では、`prefill_real_batch=true` / `decode_real_batch=true` とbatched counterを条件にする。
