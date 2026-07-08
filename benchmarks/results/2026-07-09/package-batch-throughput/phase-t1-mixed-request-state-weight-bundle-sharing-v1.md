# T1 mixed request-state weight-bundle sharing v1

## 前回の要点

- `package-token-ids-mixed-request-state-smoke` は `manifest-all` full mixed orderで通っていた。
- AQ4 payload、AQ4 scale/row-scale、主要passthrough weight bufferはrequest slot間で共有済みだった。
- ただしlayer struct内ではweight fieldとstate/workspace fieldが混在しており、weight-only resident bundle境界はまだ明示されていなかった。

## 今回の変更点

- self-attention resident layerを `PackageSelfAttnResidentStepWeights` とrequest slot state/workspaceへ分けた。
- linear-attention resident layerを `PackageLinearAttnResidentStepWeights` とrequest slot state/workspaceへ分けた。
- batch layer loaderは1slot目でweight bundleを作り、2slot目以降は同じ `Arc<...Weights>` からstate/workspaceだけを作る。
- requestごとのpaged KV cache、block table、Conv1d history、recurrent state、workspaceは引き続きslot別に残した。
- smoke出力に `self_attn_weight_bundle_shared=true` と `linear_attn_weight_bundle_shared=true` を追加した。

## R9700 smoke

Command:

```text
target/debug/ullm-engine package-token-ids-mixed-request-state-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d 2 1048576 manifest-all len:2x2 1 1 1024 32 10000000 0
```

Result:

| field | scale/passthrough shared | weight bundle shared |
| --- | ---: | ---: |
| layers | `0..31` | `0..31` |
| requests | 2 | 2 |
| final top1 tokens | `44370,5446` | `44370,5446` |
| slot AQ4 payload registry shared | `true` | `true` |
| slot AQ4 scale values shared | `true` | `true` |
| slot passthrough weight buffers shared | `true` | `true` |
| self-attn weight bundle shared | n/a | `true` |
| linear-attn weight bundle shared | n/a | `true` |
| layer load ms | 16240.561131 | 10256.735321 |
| total wall ms | 16859.363507 | 10907.692026 |
| prefill tok/s | 38.223872 | 38.077719 |
| decode tok/s | 81.553350 | 81.676422 |
| verified | `true` | `true` |

## 判断

- full mixed request-state pathで、layerごとに1つのweight-only resident bundleを作る境界は通った。
- final top1 tokenは前回のmanifest smoke、payload共有後、scale/passthrough共有後と一致した。
- `layer_load_ms` はscale/passthrough共有後からさらに約 `5983.825810 ms` 短縮した。
- これはまだreal batch throughput rowではない。実行は `batching_mode=request_state_interleaved` で、`prefill_real_batch=false` / `decode_real_batch=false` のままである。

## 次の行動

1. request-state interleaved stepをreal request-batch executorへ置き換える。
2. full packageで `batching.mode=real`、`prefill_real_batch=true`、`decode_real_batch=true` のAQ4 baseline rowを保存する。
3. T2 SQ候補を同じfull mixed pathへ接続し、AQ4/SQ比較へ進む。
