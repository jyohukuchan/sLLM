# T2 SQ FP8 qkv layer23 v4 recovery prompt bundle v1

## 前回の要点

- layer23 `k16` はprompt bundleとB=1/4/8 short guardでstrict top1を維持した。
- layer23 `q8` と `v16` は単体でも`case_a`を崩したため、scale強化で回復できるかを見る段階だった。

## 今回の変更点

- layer23 `k16` 通過branchにlayer23 `v4` を追加した。
- SQ側は `ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_TRIPLE_KERNEL=1` と `ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_KERNEL=1` でdirect kernelを必須化した。

## R9700 result

| row | FP8 tensors | single count | triple count | prefill tok/s | decode tok/s | end-to-end tok/s | VRAM consumed bytes | final top1 | strict top1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| AQ4 baseline reused | 0 | 0 | 0 | 66.811351 | 80.883171 | 35.370726 | 4443144192 | `24218,4105,329` | true |
| SQ `layer23 v4` | 17 | 46 | 115 | 59.687315 | 73.212004 | 32.586644 | 4609056768 | `24218,5582,329` | false |

## Quality comparison

| prompt | AQ4 top1 | SQ top1 | strict top1 | top8 overlap | AQ4 top1 rank in SQ top8 | SQ top1 margin over AQ4 top1 | SQ top1 margin over rank2 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `len4` | 24218 | 24218 | pass | 7 / 8 | 1 | 0.000000000 | 0.298209667 |
| `case_a` | 4105 | 5582 | fail | 8 / 8 | 2 | 0.000437260 | 0.000437260 |
| `case_b` | 329 | 329 | pass | 8 / 8 | 1 | 0.000000000 | 0.208777905 |

## 判断

- layer23 `v4` は`case_a`でstrict top1を落とした。
- AQ4 top1 `4105` はSQ top8内の2位に残るが、現T2 promotion ruleはstrict top1なので失敗として扱う。
- short batch guardへは進めない。

Artifacts:

- `results.schema.jsonl`
- `results.jsonl`
- `comparison.json`
- raw/stdout/stderr/memory logs

## 次の行動

1. このscale候補をfailure guardとして保存する。
2. layer23 q/vを回復するには、row-block幅だけでなく別format/layoutまたはtext-level guardを検討する。
