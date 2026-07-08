# T1 self-attn stack real-batch small grid v1

## 前回の要点

- `phase-t1-token-id-model-loop-hybrid-smoke-v1` では、token ID inputからselected-layer model-loopとlm_head top1 guardまでは接続できていた。
- その時点ではprefillがrequest-batch化されておらず、`batching.mode=hybrid`、`prefill_real_batch=false` として保存されていた。
- full packageにはself-attention層とlinear-attention層が混在しており、既存のtoken-id model-loop runnerはfull mixed-attention layer orderを直接は実行できなかった。

## 今回の変更点

- `package-token-ids-model-loop-smoke` と `sq-fp8-token-ids-model-loop-smoke` に `all-self-attn` layer aliasを追加した。
- `all-self-attn` はpackage manifestのself-attention `q_norm` / `k_norm` passthrough tensor集合からlayer indexを抽出し、昇順のself-attention stackとして実行する。
- `all` はmixed-attention full packageを推定できないため、明示的に拒否する。full mixed-attention throughputにはlinear-attention層を含む別runnerがまだ必要である。
- R9700でAQ4 packageのmanifest self-attention 8層 `3,7,11,15,19,23,27,31` を、`batch=1/4/8`、`prompt=4`、`generated=1` で測定した。

## 実測値

この表はmanifest self-attention層だけを通した中間model-loop rowであり、Qwen3.5-9B full mixed-attention LM throughputではない。

| case | batching | prefill real batch | decode real batch | prefill request parallelism | decode request parallelism | prefill total input tok/s | decode generated tok/s | end-to-end tok/s | VRAM consumed bytes | final top1 tokens |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `B=1, pp=4, tg=1` | `hybrid` | false | false | 1 | 1 | 74.066673 | 70.654751 | 73.358179 | 7435599872 | `208306` |
| `B=4, pp=4, tg=1` | `real` | true | true | 4 | 4 | 73.537780 | 71.298348 | 73.078709 | 7435612160 | `208306,315,140864,136706` |
| `B=8, pp=4, tg=1` | `real` | true | true | 8 | 8 | 73.326934 | 71.010893 | 72.851718 | 7563571200 | `208306,315,140864,136706,151353,140864,180678,180678` |

## 判断

- `batch=4/8` では `prefill_real_batch=true` と `decode_real_batch=true` が保存され、request-batch prefill/decode pathはself-attention stack上で動作した。
- この短い `prompt=4` 条件では、B=1からB=8へ増やしてもtotal tok/sはほぼ伸びなかった。少なくともこの中間rowでは、batch化の効果よりself-attention stackの固定実行コストと短いprompt長の影響が強いと考える。
- これはfull mixed-attention runnerではない。linear-attention層、full layer order、full package KV/cache accountingを含むreal-batch end-to-end rowは未完了である。
- `all-self-attn` aliasは、mixed package内のself-attention subsetを安全に選ぶための中間手段として有効である。

## Artifacts

- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t1-self-attn-stack-real-batch-small-grid-v1/results.jsonl`
- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t1-self-attn-stack-real-batch-small-grid-v1/aq4-r9700-selfattnstack-real-b1-pp4-tg1/raw.json`
- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t1-self-attn-stack-real-batch-small-grid-v1/aq4-r9700-selfattnstack-real-b4-pp4-tg1/raw.json`
- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t1-self-attn-stack-real-batch-small-grid-v1/aq4-r9700-selfattnstack-real-b8-pp4-tg1/raw.json`

## 次の行動

1. `all-self-attn` rowはT1の中間real-batch guardとして扱い、SQ performance decision用のfull LM throughputとは扱わない。
2. 次のT1本命は、linear-attention層を含むfull mixed-attention package real-batch prefill/decode/end-to-end runnerである。
3. そのrunnerでは `batch=1/4/8`、prefill/decode/end-to-end total throughput、VRAM、KV cache bytes、quality guardを同じ `inference-benchmark-result-v0.1` schemaへ保存する。
