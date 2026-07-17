# T1 package prefill component batch grid

## 前回の要点

- `.ullm.d` package由来のprefill component smokeをJSONLへ保存できるrunnerを追加した。
- 初回smokeは`batch_size=1`で、full package total throughputではなかった。

## 今回の変更点

- `tools/run-package-prefill-component-workload.py` に `component_args_template` を追加した。
- `prompt_tokens * concurrent_requests` を `component_total_prompt_tokens` として展開し、projection componentでは `len:{component_total_prompt_tokens}` でflattened token-parallel実行できるようにした。
- `run-external-benchmark.py --parse ullm-component-prefill` を調整し、package component reportに`batch_count`が無い場合はworkload側の`batch_size`と`concurrent_requests`を保持するようにした。
- R9700でAQ4 package `k_proj` componentを `batch=1,prompt=2` と `batch=4,prompt=2` で測った。
- B=4 caseは `workload.batch_size=4`、`prompt_tokens_per_request=[2,2,2,2]`、`component_total_input_tokens=8`、`prefill_executor_token_parallelism=8`、`prefill_executor_request_parallelism=1` として保存された。
- 結果は `benchmarks/results/2026-07-08/package-batch-throughput/phase-t1-package-prefill-component-batch-grid-v1.*` に保存した。

## 次の行動

1. request boundaryが効くself-attention layer componentへ広げる。
2. full package total throughputにはdecode/end-to-end runnerがまだ必要。
3. SQ throughput判断では、このcomponent gridを最終速度として使わない。
