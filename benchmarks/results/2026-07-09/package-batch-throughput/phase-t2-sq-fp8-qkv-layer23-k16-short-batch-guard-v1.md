# T2 SQ FP8 qkv layer23 k16 short batch guard v1

## 前回の要点

- layer23 split prompt bundleでは、`k16` だけがstrict top1 `3 / 3` を維持した。
- ただし`case_a`のmarginは非常に薄いため、短いbatch guardで崩れないか確認する必要があった。

## 今回の変更点

- `sq-fp8-w8a16-r9700-v0-qkv-layers3-7-11-15-19-q8-k16-v16-plus-layer23-k16` をB=1/4/8のshort guardへ流した。
- promptは `len:2xB`、generated tokenは1、lm_head top_kは1にした。
- AQ4 baselineは `phase-t2-sq-fp8-qkv-q16-k16-v16-short-batch-guard-v1` の同一workload結果を再利用した。
- SQ側は `ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_TRIPLE_KERNEL=1` と `ULLM_REQUIRE_HIP_SQ_FP8_MATVEC_KERNEL=1` でtriple/single direct kernelを必須化した。

## R9700 result

| B | AQ4 prefill tok/s | SQ prefill tok/s | AQ4 decode tok/s | SQ decode tok/s | AQ4 end-to-end tok/s | SQ end-to-end tok/s | top1 match | SQ single count | SQ triple count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 1 | 24.997350 | 17.649348 | 80.313241 | 70.880232 | 10.012651 | 8.779914 | true | 3 | 15 |
| 4 | 53.250215 | 48.840512 | 81.519502 | 73.523424 | 26.309008 | 24.964220 | true | 12 | 60 |
| 8 | 65.369008 | 60.051932 | 81.728132 | 73.442133 | 36.593422 | 34.338277 | true | 24 | 120 |

Final top1:

| B | AQ4 | SQ |
| ---: | --- | --- |
| 1 | `44370` | `44370` |
| 4 | `44370,5446,10701,25411` | `44370,5446,10701,25411` |
| 8 | `44370,5446,10701,25411,21901,685,279,27973` | `44370,5446,10701,25411,21901,685,279,27973` |

## 判断

- layer23 `k16` はB=1/4/8のshort guardでもstrict top1を維持した。
- SQ FP8 direct pathは、既存5層QKVのtriple境界に加えてlayer23 `k_proj` のsingle direct境界を踏んでいる。
- layer23 `q8` と `v16` はprompt bundleで単体失敗しているため、現時点ではlayer23 `k16`だけをdiagnostic extensionとして扱う。
- prompt bundleの`case_a` marginが薄いため、まだfull SQ policyとしてpromoteしない。

Artifacts:

- `results.schema.jsonl`
- `results.jsonl`
- `comparison.json`
- `sq-qkv-layers3-7-11-15-19-plus-layer23-k16-b*/raw.json`

## 次の行動

1. layer23 `k16` をcurrent diagnostic extensionとして維持する。
2. layer23 `q_proj` と `v_proj` はfallbackまたは別scale/layout候補として再探索する。
3. 次はlayer27のk-only extension、またはlayer23 q/vのscale強化、または広いprompt/text guardへ進む。
