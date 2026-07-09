# T2 SQ FP8 full mixed qkv v16 prompt bundle v1

## 前回の要点

- layer3 `q/k/v` の `q32/k16/v32` triple候補は、full mixed prompt bundleで `case_a` のtop1をAQ4 `4105` からSQ `5582` に入れ替えた。
- `q/k` pair単体と `v` 単体は同じprompt bundleでstrict top1を維持した。
- 次の仮説は、`q/k/v` 同時適用時の累積driftを、`v_proj` のscale粒度を細かくすることで戻せるかだった。

## 今回の変更点

- `q32/k16/v16` policy `sq-fp8-w8a16-r9700-v0-qkv-layer3-q32-k16-v16` を追加した。
- `v_proj` をrow-block32からrow-block16に変更し、`q_proj` はrow-block32、`k_proj` はrow-block16のままにした。
- artifactは `/tmp/ullm-sq-fp8-qkv-layer3-q32-k16-v16-policy-v0.1-artifact` に生成した。
- `ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_TRIPLE_KERNEL=1` を付け、triple direct SQ FP8 kernelが踏まれることをtelemetryで確認した。

## R9700 result

| row | prefill real | decode real | prefill tok/s | decode tok/s | end-to-end tok/s | VRAM consumed bytes | final top1 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| AQ4 baseline | true | true | 67.241771 | 81.154855 | 35.047795 | 4443152384 | `24218,4105,329` |
| SQ `q32/k16/v16` | true | true | 62.908898 | 79.445748 | 34.061110 | 4975894528 | `24218,5582,329` |

SQ telemetry:

| field | value |
| --- | ---: |
| `sq_projection_boundary` | `triple` |
| `sq_fp8_triple_matvec_count` | 23 |

## Quality comparison

| prompt | AQ4 top1 | SQ top1 | strict top1 | top8 overlap | AQ4 top1 rank in SQ top8 | note |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| `len4` | 24218 | 24218 | pass | 7 / 8 | 1 | top1 stable |
| `case_a` | 4105 | 5582 | fail | 8 / 8 | 2 | SQ top1 margin over AQ4 top1 is `0.000260353` |
| `case_b` | 329 | 329 | pass | 8 / 8 | 1 | top1 stable |

## 判断

- `v_proj` をrow-block16に細かくしても、`case_a` のtop1 swapは戻らなかった。
- ただしfailure marginは `q32/k16/v32` の約 `0.001122474` から `0.000260353` へ縮んだ。
- `v16` は方向としては改善しているが、strict top1 guardを通すには不十分だった。
- 次は `q_proj` もrow-block16へ細かくした `q16/k16/v16` を試す。

Artifacts:

- `benchmarks/results/2026-07-09/sq-fp8-qkv-layer3-q32-k16-v16-policy-v0.1.json`
- `benchmarks/results/2026-07-09/sq-fp8-qkv-layer3-q32-k16-v16-policy-artifact-v0.1.json`
- `results.schema.jsonl`
- `results.jsonl`
- `comparison.json`

## 次の行動

1. `q32/k16/v16` はfailure boundaryとして保存する。
2. `q16/k16/v16` のfull mixed prompt bundle結果を優先して見る。
