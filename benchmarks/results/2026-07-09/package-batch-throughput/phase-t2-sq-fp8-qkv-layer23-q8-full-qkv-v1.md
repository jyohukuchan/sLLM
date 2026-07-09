# T2 SQ FP8 qkv layer23 q8 full QKV prompt bundle v1

## 前回の要点

- layer19 `q8/k16/v16` はfull mixed prompt bundleとB=1/4/8 short guardでstrict top1を維持した。
- 次の境界として、同じQKV patternをlayer23へ広げられるか確認する必要があった。

## 今回の変更点

- `sq-fp8-w8a16-r9700-v0-qkv-layers3-7-11-15-19-23-q8-k16-v16` を作成した。
- layer23は `q_proj` をrow-block8、`k_proj/v_proj` をrow-block16にした。
- SQ側は `ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_TRIPLE_KERNEL=1` と `ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_KERNEL=1` でdirect kernelを必須化した。

## R9700 result

| row | FP8 tensors | single count | triple count | prefill tok/s | decode tok/s | end-to-end tok/s | VRAM consumed bytes | final top1 | strict top1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| AQ4 baseline reused | 0 | 0 | 0 | 66.811351 | 80.883171 | 35.370726 | 4443144192 | `24218,4105,329` | true |
| SQ `layers3/7/11/15/19/23 q8/k16/v16` | 18 | 0 | 138 | 60.821302 | 72.084189 | 33.395723 | 5275955200 | `24218,5582,329` | false |

## Quality comparison

| prompt | AQ4 top1 | SQ top1 | strict top1 | top8 overlap | AQ4 top1 rank in SQ top8 | SQ top1 margin over AQ4 top1 | SQ top1 margin over rank2 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `len4` | 24218 | 24218 | pass | 7 / 8 | 1 | 0.000000000 | 0.300285816 |
| `case_a` | 4105 | 5582 | fail | 8 / 8 | 2 | 0.001451493 | 0.001451493 |
| `case_b` | 329 | 329 | pass | 8 / 8 | 1 | 0.000000000 | 0.210827351 |

## 判断

- layer23 full QKV extensionは`case_a`でstrict top1を落とした。
- `case_a`ではAQ4 top1 `4105` がSQ top8内の2位に残り、SQ top1 `5582` との差は小さいが、現T2 promotion ruleはstrict top1なので失敗として扱う。
- full QKV candidateはshort batch guardへ進めず、layer23 q/k/v splitで原因を切り分ける。

Artifacts:

- `results.schema.jsonl`
- `results.jsonl`
- `comparison.json`
- `sq-qkv-layers3-7-11-15-19-23-q8-k16-v16/raw.json`

## 次の行動

1. layer23 q/k/v split結果を確認する。
2. qまたはvが単体で失敗する場合は、そのfamilyをfallbackまたはより強いscale候補へ戻す。
3. passしたsplit候補だけをshort batch guardへ進める。
