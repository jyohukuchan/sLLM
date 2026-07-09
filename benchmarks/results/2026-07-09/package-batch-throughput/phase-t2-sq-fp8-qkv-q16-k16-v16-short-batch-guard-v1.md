# T2 SQ FP8 qkv q16/k16/v16 short batch guard v1

## 前回の要点

- layer3 `q16/k16/v16` はfull mixed prompt bundleでstrict top1 `3 / 3` を維持した。
- 同じscale粒度をlayer7以降へ広げる前に、短いB=1/4/8 guardで再確認する必要があった。

## 今回の変更点

- `sq-fp8-w8a16-r9700-v0-qkv-layer3-q16-k16-v16` をB=1/4/8のshort guardへ流した。
- promptは `len:2xB`、generated tokenは1、lm_head top_kは1にした。
- AQ4 baselineとSQ FP8候補を同じrun directoryへ保存した。
- SQ側は `ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_TRIPLE_KERNEL=1` でtriple direct kernelを必須化した。

## R9700 result

| B | AQ4 prefill tok/s | SQ prefill tok/s | AQ4 decode tok/s | SQ decode tok/s | AQ4 end-to-end tok/s | SQ end-to-end tok/s | top1 match | SQ triple count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 1 | 24.997350 | 20.565753 | 80.313241 | 79.245965 | 10.012651 | 9.414972 | true | 3 |
| 4 | 53.250215 | 48.680545 | 81.519502 | 79.959365 | 26.309008 | 24.906282 | true | 12 |
| 8 | 65.369008 | 61.753734 | 81.728132 | 80.030092 | 36.593422 | 35.576634 | true | 24 |

Final top1:

| B | AQ4 | SQ |
| ---: | --- | --- |
| 1 | `44370` | `44370` |
| 4 | `44370,5446,10701,25411` | `44370,5446,10701,25411` |
| 8 | `44370,5446,10701,25411,21901,685,279,27973` | `44370,5446,10701,25411,21901,685,279,27973` |

## 判断

- `q16/k16/v16` はB=1/4/8のshort guardでもstrict top1を維持した。
- SQ FP8 direct triple境界はB=1/4/8でそれぞれ `3/12/24` 回踏まれている。
- full mixed prompt bundleとshort batch guardの両方を通ったため、layer3 QKV triple pass boundaryとしての信頼度は上がった。
- ただし対象はlayer3の3 tensorだけなので、full SQ policyではない。

Artifacts:

- `results.schema.jsonl`
- `results.jsonl`
- `comparison.json`
- `aq4-b*/raw.json`
- `sq-qkv-q16-k16-v16-b*/raw.json`

## 次の行動

1. `q16/k16/v16` をlayer3 QKV triple pass boundaryとして固定する。
2. 次は同じrow-block16をlayer7のQKVへ足した候補を作り、full mixed prompt bundleでstrict top1を確認する。
3. 追加layerで崩れる場合は、layer7のQ/K/Vを単体またはpairで分解する。
