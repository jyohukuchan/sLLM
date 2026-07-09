# T2 SQ FP8 qkv layer23 split prompt bundle v1

## 前回の要点

- layer23 `q8/k16/v16` full QKV extensionは`case_a`でAQ4 top1 `4105` からSQ top1 `5582` に入れ替わり、strict top1に失敗した。
- short batch guardへ進む前に、layer23の `q_proj`、`k_proj`、`v_proj` を個別に切り分ける必要があった。

## 今回の変更点

- layer19 `q8/k16/v16` 通過boundaryをbaseにして、layer23の `q_proj`、`k_proj`、`v_proj` を1つずつ追加した。
- `q_proj` はrow-block8、`k_proj/v_proj` はrow-block16で評価した。
- SQ側は `ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_TRIPLE_KERNEL=1` と `ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_KERNEL=1` を使い、既存層のtriple directとlayer23単体のsingle directを必須化した。

## R9700 result

| row | FP8 tensors | single count | triple count | prefill tok/s | decode tok/s | end-to-end tok/s | VRAM consumed bytes | final top1 | strict top1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| AQ4 baseline reused | 0 | 0 | 0 | 66.811351 | 80.883171 | 35.370726 | 4443144192 | `24218,4105,329` | true |
| SQ `q8` | 16 | 23 | 115 | 59.189428 | 72.379743 | 32.133131 | 5141766144 | `24218,5582,329` | false |
| SQ `k16` | 16 | 23 | 115 | 59.818278 | 72.952135 | 33.050611 | 5114462208 | `24218,4105,329` | true |
| SQ `v16` | 16 | 23 | 115 | 59.042268 | 72.719037 | 32.798374 | 5242413056 | `24218,5582,329` | false |

## Quality comparison

### q8

| prompt | AQ4 top1 | SQ top1 | strict top1 | top8 overlap | AQ4 top1 rank in SQ top8 | SQ top1 margin over rank2 |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `len4` | 24218 | 24218 | pass | 7 / 8 | 1 | 0.286063194 |
| `case_a` | 4105 | 5582 | fail | 8 / 8 | 2 | 0.000849724 |
| `case_b` | 329 | 329 | pass | 8 / 8 | 1 | 0.203430176 |

### k16

| prompt | AQ4 top1 | SQ top1 | strict top1 | top8 overlap | AQ4 top1 rank in SQ top8 | SQ top1 margin over rank2 |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `len4` | 24218 | 24218 | pass | 7 / 8 | 1 | 0.291079521 |
| `case_a` | 4105 | 4105 | pass | 8 / 8 | 1 | 0.000099182 |
| `case_b` | 329 | 329 | pass | 8 / 8 | 1 | 0.201835633 |

### v16

| prompt | AQ4 top1 | SQ top1 | strict top1 | top8 overlap | AQ4 top1 rank in SQ top8 | SQ top1 margin over rank2 |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `len4` | 24218 | 24218 | pass | 7 / 8 | 1 | 0.278463841 |
| `case_a` | 4105 | 5582 | fail | 8 / 8 | 2 | 0.000475406 |
| `case_b` | 329 | 329 | pass | 8 / 8 | 1 | 0.209313869 |

## 判断

- layer23 `k16` だけがprompt bundleでstrict top1 `3 / 3` を維持した。
- layer23 `q8` と `v16` は単体追加でも`case_a`が `4105 -> 5582` に反転するため、full QKV failureの主因候補としてfallbackに戻す。
- layer23 `k16` も`case_a` marginが非常に薄いため、B=1/4/8 short guardで追加確認する。

Artifacts:

- `results.schema.jsonl`
- `results.jsonl`
- `comparison.json`
- `sq-qkv-layers3-7-11-15-19-plus-layer23-{q8,k16,v16}/raw.json`

## 次の行動

1. layer23 `k16` をshort batch guardへ進める。
2. layer23 `q_proj` と `v_proj` はfallback維持、またはrow-block幅/scale layoutの別候補として再探索する。
3. full SQ policyへpromoteする前にtext-level guardを追加する。
