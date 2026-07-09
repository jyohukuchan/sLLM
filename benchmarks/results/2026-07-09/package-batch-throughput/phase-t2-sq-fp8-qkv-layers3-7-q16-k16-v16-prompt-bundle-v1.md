# T2 SQ FP8 qkv layers3/7 q16/k16/v16 prompt bundle v1

## 前回の要点

- layer3 `q16/k16/v16` はfull mixed prompt bundleでstrict top1 `3 / 3` を維持した。
- 同じ候補はB=1/4/8 short guardでもAQ4 top1を維持し、SQ FP8 triple boundaryを踏むことを確認した。
- 次は同じrow-block16 QKVをlayer7へ足して、full mixed prompt bundleで崩れるかを見る段階だった。

## 今回の変更点

- `sq-fp8-w8a16-r9700-v0-qkv-layers3-7-q16-k16-v16` policyを追加した。
- layer3とlayer7の `q_proj`、`k_proj`、`v_proj` をすべてrow-block16 FP8にした。
- artifactは `/tmp/ullm-sq-fp8-qkv-layers3-7-q16-k16-v16-policy-v0.1-artifact` に生成した。
- R9700でfull mixed `manifest-all` prompt bundleを実行し、AQ4 baselineとSQ FP8候補を比較した。
- SQ側は `ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_TRIPLE_KERNEL=1` でtriple direct kernelを必須化した。

## R9700 result

| row | FP8 tensors | prefill real | decode real | prefill tok/s | decode tok/s | end-to-end tok/s | VRAM consumed bytes | final top1 |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| AQ4 baseline | 0 | true | true | 66.811351 | 80.883171 | 35.370726 | 4443144192 | `24218,4105,329` |
| SQ `layers3/7 q16/k16/v16` | 6 | true | true | 62.866188 | 77.707480 | 34.095768 | 4499853312 | `24218,4105,329` |

SQ telemetry:

| field | value |
| --- | ---: |
| `sq_projection_boundary` | `triple` |
| `sq_fp8_triple_matvec_count` | 46 |
| `sq_passthrough_tensor_count` | 769 |

## Quality comparison

| prompt | AQ4 top1 | SQ top1 | strict top1 | top8 overlap | AQ4 top1 rank in SQ top8 | SQ top1 margin over rank2 |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `len4` | 24218 | 24218 | pass | 7 / 8 | 1 | 0.297621727 |
| `case_a` | 4105 | 4105 | pass | 8 / 8 | 1 | 0.003049851 |
| `case_b` | 329 | 329 | pass | 8 / 8 | 1 | 0.187139512 |

## 判断

- layer3+7 `q16/k16/v16` はfull mixed prompt bundleでstrict top1 `3 / 3` を維持した。
- SQ FP8 direct triple境界も `sq_fp8_triple_matvec_count=46` で確認できた。
- `case_a` のmarginは薄いが、layer3単独の `0.002023697` から `0.003049851` へ少し広がった。
- ただし2 self-attention layer / 6 tensorだけの診断候補なので、full SQ policyではない。
- 現在のQKV triple passing boundaryは layer3+7 `q16/k16/v16` として扱う。

Artifacts:

- `benchmarks/results/2026-07-09/sq-fp8-qkv-layers3-7-q16-k16-v16-policy-v0.1.json`
- `benchmarks/results/2026-07-09/sq-fp8-qkv-layers3-7-q16-k16-v16-policy-artifact-v0.1.json`
- `results.schema.jsonl`
- `results.jsonl`
- `comparison.json`

## 次の行動

1. layer3+7 `q16/k16/v16` を現在のQKV triple passing boundaryとして保存する。
2. 次はlayer11 QKVを同じrow-block16で足し、full mixed prompt bundleでstrict top1を確認する。
3. layer11追加で崩れる場合は、layer11のQ/K/Vを単体またはpairで分解する。
