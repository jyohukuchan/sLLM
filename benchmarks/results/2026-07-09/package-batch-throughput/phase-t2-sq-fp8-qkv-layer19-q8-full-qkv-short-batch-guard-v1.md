# T2 SQ FP8 qkv layer19 q8 full QKV short batch guard v1

## 前回の要点

- layer19 `q8/k16/v16` はfull mixed prompt bundleでstrict top1 `3 / 3` を維持した。
- ただし `case_a` のSQ top1 marginは `0.000214577` と薄く、短いbatch guardで確認する必要があった。

## 今回の変更点

- `sq-fp8-w8a16-r9700-v0-qkv-layers3-7-11-15-19-q8-k16-v16` をB=1/4/8のshort guardへ流した。
- promptは `len:2xB`、generated tokenは1、lm_head top_kは1にした。
- AQ4 baselineは `phase-t2-sq-fp8-qkv-q16-k16-v16-short-batch-guard-v1` の同一workload結果を再利用した。
- SQ側は `ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_TRIPLE_KERNEL=1` でtriple direct kernelを必須化した。

## R9700 result

| B | AQ4 prefill tok/s | SQ prefill tok/s | AQ4 decode tok/s | SQ decode tok/s | AQ4 end-to-end tok/s | SQ end-to-end tok/s | top1 match | SQ triple count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 1 | 24.997350 | 19.387802 | 80.313241 | 71.386978 | 10.012651 | 8.774435 | true | 15 |
| 4 | 53.250215 | 49.981944 | 81.519502 | 74.061735 | 26.309008 | 24.948615 | true | 60 |
| 8 | 65.369008 | 59.417716 | 81.728132 | 73.850648 | 36.593422 | 34.272665 | true | 120 |

Final top1:

| B | AQ4 | SQ |
| ---: | --- | --- |
| 1 | `44370` | `44370` |
| 4 | `44370,5446,10701,25411` | `44370,5446,10701,25411` |
| 8 | `44370,5446,10701,25411,21901,685,279,27973` | `44370,5446,10701,25411,21901,685,279,27973` |

## 判断

- layer19 `q8/k16/v16` はB=1/4/8のshort guardでもstrict top1を維持した。
- SQ FP8 direct triple境界はB=1/4/8でそれぞれ `15/60/120` 回踏まれている。
- prefill/decode tok/sはAQ4 baselineより低いが、今回は品質境界とtriple path疎通のguardであり、速度採用判定ではない。
- prompt bundleの`case_a` marginが薄いため、まだfull SQ policyとしてpromoteしない。

Artifacts:

- `results.schema.jsonl`
- `results.jsonl`
- `comparison.json`
- `sq-qkv-layers3-7-11-15-19-q8-k16-v16-b*/raw.json`

## 次の行動

1. layer19 `q8/k16/v16` をcurrent diagnostic recovery candidateとして維持する。
2. 次はlayer23 QKV extension、または広いprompt/text guardで薄いmarginが崩れないか確認する。
3. full SQ policyへpromoteする前に、strict top1だけでなくtext-level guardを追加する。
