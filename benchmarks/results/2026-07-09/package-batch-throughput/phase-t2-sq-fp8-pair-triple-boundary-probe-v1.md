# T2 SQ FP8 pair/triple boundary probe v1

## 前回の要点

- SQ FP8 direct pathは単発 `matvec`、`matvec_batch`、`matvec_pair_with`、`matvec_triple_with` まで実装済みだった。
- ただしfull mixed strict-top1保守候補はlayer3 `k_proj` 1 tensorだけで、実ベンチではpair/triple境界を踏めていなかった。
- 次のT2作業は、`q/k` または `q/k/v` が同時にstrict-top1を維持できる最小候補を作り、full mixed rowとして保存することだった。

## 今回の変更点

- layer3 `q/k` 候補 `sq-fp8-w8a16-r9700-v0-qk-layer3-q32-k16` を追加した。
- layer3 `q/k/v` 候補 `sq-fp8-w8a16-r9700-v0-qkv-layer3-q32-k16-v32` を追加した。
- `q_proj` と `v_proj` はrow-block32、`k_proj` はrow-block16でartifactを生成した。
- `q/k` は `ULLM_DISABLE_AQ4_MATVEC_TRIPLE_SELF_ATTN_QKV=1` を指定し、q/k pair dispatchを踏む条件でB=1/4/8を測った。
- `q/k/v` は通常のself-attention q/k/v triple dispatchでB=1/4/8を測った。

## Artifacts

| candidate | artifact | FP8 tensors | passthrough tensors | compact resident bytes estimate |
| --- | --- | ---: | ---: | ---: |
| `qk-layer3-q32-k16` | `/tmp/ullm-sq-fp8-qk-layer3-q32-k16-policy-v0.1-artifact` | 2 | 773 | 19273710560 |
| `qkv-layer3-q32-k16-v32` | `/tmp/ullm-sq-fp8-qkv-layer3-q32-k16-v32-policy-v0.1-artifact` | 3 | 772 | 19270040544 |

Policy/artifact summaries:

- `benchmarks/results/2026-07-09/sq-fp8-qk-layer3-q32-k16-policy-v0.1.json`
- `benchmarks/results/2026-07-09/sq-fp8-qk-layer3-q32-k16-policy-artifact-v0.1.json`
- `benchmarks/results/2026-07-09/sq-fp8-qkv-layer3-q32-k16-v32-policy-v0.1.json`
- `benchmarks/results/2026-07-09/sq-fp8-qkv-layer3-q32-k16-v32-policy-artifact-v0.1.json`

## R9700 full mixed rows

Common package:

```text
/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d
```

Common workload:

```text
manifest-all len:2xB generated=1 top_k=1 lm_head_chunk_rows=1024 rotary_dim=32 rope_base=10000000 position_offset=0
```

| boundary | batch | mode | prefill real | decode real | FP8 tensors | SQ prefill tok/s | SQ decode tok/s | SQ end-to-end tok/s | AQ4 end-to-end tok/s | final top1 | top1 match | VRAM consumed bytes |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| q/k pair | 1 | `single` | `false` | `false` | 2 | 20.093980 | 77.380772 | 9.324354 | 8.926325 | `44370` | `true` | 3948146688 |
| q/k pair | 4 | `real` | `true` | `true` | 2 | 16.546983 | 78.429273 | 15.277466 | 24.096096 | `44370,5446,10701,25411` | `true` | 4522790912 |
| q/k pair | 8 | `real` | `true` | `true` | 2 | 58.769493 | 78.645271 | 34.187742 | 34.577530 | `44370,5446,10701,25411,21901,685,279,27973` | `true` | 4782911488 |
| q/k/v triple | 1 | `single` | `false` | `false` | 3 | 17.564031 | 78.859963 | 8.414481 | 8.926325 | `44370` | `true` | 4210311168 |
| q/k/v triple | 4 | `real` | `true` | `true` | 3 | 48.433379 | 80.057414 | 25.402500 | 24.096096 | `44370,5446,10701,25411` | `true` | 5034610688 |
| q/k/v triple | 8 | `real` | `true` | `true` | 3 | 63.183079 | 79.943546 | 35.980289 | 34.577530 | `44370,5446,10701,25411,21901,685,279,27973` | `true` | 4784947200 |

All rows:

- `status=ok`
- `verified=true`
- `sq_overlay=true`
- `sq_execution_mode=direct_fp8_dequant_matvec`
- `throughput_row=true`
- `load_excluded_from_total=true`
- `final_logits_in_total=true`
- `request_batch_executor=true`
- `fused_request_batch=false`
- `prefill_mode=token_id_full_mixed_request_state`
- `layers_csv=0..31`

Pair-specific guards:

- `ULLM_DISABLE_AQ4_MATVEC_TRIPLE_SELF_ATTN_QKV=1`
- `ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_PAIR_KERNEL=1`

Triple-specific guards:

- `ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_TRIPLE_KERNEL=1`

Raw artifacts:

- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-pair-triple-boundary-probe-v1/results.schema.jsonl`
- `benchmarks/results/2026-07-09/package-batch-throughput/phase-t2-sq-fp8-pair-triple-boundary-probe-v1/results.jsonl`
- per-row `raw.json`, `stdout.log`, `stderr.log`, and `memory.jsonl`

## 判断

- `q/k` pair候補はB=1/4/8でAQ4 final top1と一致した。ただしpair検証のためにq/k/v tripleを無効化しており、vはAQ4単発matvecなので、速度比較は通常self-attention pathそのものではない。
- `q/k/v` triple候補もB=1/4/8でAQ4 final top1と一致した。これはSQ FP8 triple境界を踏む最小full mixed候補として扱える。
- B=4/B=8では `q/k/v` triple候補のend-to-end tok/sがAQ4 baselineを少し上回った。ただしFP8 tensor数は3個だけで、promptは `len:2xB` の短いsmokeなので、SQ format性能の結論にはまだ使わない。
- stdout上の `sq_execution_mode` はまだpair/tripleを区別せず `direct_fp8_dequant_matvec` と出る。今回の境界証拠は実行条件とcandidate selectionに依存しているため、次はstdout/schemaへ `sq_projection_boundary=pair|triple|single` のような明示列を追加したほうがいい。

## 次の行動

1. `q/k/v` layer3を最小triple boundary候補として固定し、prompt bundleまたは長めのprefill gridでqualityを再確認する。
2. stdout/JSONLにSQ pair/triple境界の実行モードを明示し、将来の速度表で単発/pair/tripleを混同しないようにする。
3. 次の候補探索ではlayer7以降の `q/k/v`、または既存の `k/o/down` branchへtriple候補を安全に足せるかを見る。
