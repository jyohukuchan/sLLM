# T2 SQ FP8 qkv layer27 k16 prompt bundle v1

## 前回の要点

- layer23 `k16` はprompt bundleとB=1/4/8 short guardでstrict top1を維持した。
- 次の境界として、layer27 `k_proj` を追加できるか確認する段階だった。

## 今回の変更点

- layer23 `k16` 通過boundaryにlayer27 `k16` を追加した。
- SQ側は `ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_TRIPLE_KERNEL=1` と `ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_KERNEL=1` でdirect kernelを必須化した。

## R9700 result

| row | FP8 tensors | single count | triple count | prefill tok/s | decode tok/s | end-to-end tok/s | VRAM consumed bytes | final top1 | strict top1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| AQ4 baseline reused | 0 | 0 | 0 | 66.811351 | 80.883171 | 35.370726 | 4443144192 | `24218,4105,329` | true |
| SQ `layer27 k16` | 17 | 46 | 115 | 59.984624 | 72.659910 | 33.269703 | 5244506112 | `24218,5582,329` | false |

## Quality comparison

| prompt | AQ4 top1 | SQ top1 | strict top1 | top8 overlap | AQ4 top1 rank in SQ top8 | SQ top1 margin over AQ4 top1 | SQ top1 margin over rank2 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `len4` | 24218 | 24218 | pass | 8 / 8 | 1 | 0.000000000 | 0.301417351 |
| `case_a` | 4105 | 5582 | fail | 8 / 8 | 2 | 0.001172543 | 0.001172543 |
| `case_b` | 329 | 329 | pass | 8 / 8 | 1 | 0.000000000 | 0.202974320 |

## 判断

- layer27 `k16` は`case_a`でstrict top1を落とした。
- AQ4 top1 `4105` はSQ top8内の2位に残るが、現T2 promotion ruleはstrict top1なので失敗として扱う。
- short batch guardへは進めず、layer27 `k_proj` はfallbackまたは別format候補へ戻す。

Artifacts:

- `results.schema.jsonl`
- `results.jsonl`
- `comparison.json`
- raw/stdout/stderr/memory logs

## 次の行動

1. layer27 `k_proj` をfailure guardとして保存する。
2. layer27以降を広げる前に、text-level guard、またはlayer23 q/vとlayer27 kの別formatを検討する。
