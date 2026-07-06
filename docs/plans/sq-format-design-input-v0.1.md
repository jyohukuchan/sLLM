# SQ Format Design Input v0.1

## 前回の要点

- Qwen3.5-9B accepted packageは、R9700/RDNA4とV620/RDNA2の両方で `prompt_tokens=512`, `generated_tokens=256` を完走した。
- decodeはR9700で `0.141 tok/s`、V620で `0.139 tok/s` で、GPU差より現行runtime経路の律速が支配的だった。
- KV cacheは約48 MiBであり、この条件のVRAM消費は主にf32 materialized weights/runtime buffersによる。

## 今回の変更点

- materialized-AQ baselineはR9700で `0.140 tok/s` となり、accepted packageと同じdecode律速を示した。
- 真のBF16 baselineは現行package/runtimeでは作れないため、pre-sq範囲ではdeferした。
- R9700/V620で短いgolden prefix reference guardを実行し、accepted packageが既存fixtureに対してverifiedになることを確認した。
- 長いdecode runは、現行経路では追加情報が少ないため、runtimeまたはsq候補が変わるまで繰り返さない方針にした。

## 次の行動

1. sq format v0.1の設計では、compact resident storageとbounded materialized working setを最優先にする。
2. runtime計測では、長いprefill圧力と短いdecode probeを分ける。
3. sq候補のrun recordには、resident bytes、materialized bytes、materialization time、prefill TPS、decode TPSを必ず入れる。

## Inputs

Primary package:

```text
/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-qwen35-hidden3994-v1-row-scale-layer6-layer10.ullm.d
```

Baseline comparison package:

```text
/tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p46-inproj-row-scale-layer6-layer10.ullm.d
```

Golden prefix fixture:

```text
benchmarks/golden/2026-07-05/qwen35-9b-prefix0-12-seq16
```

## Measured Facts

| condition | R9700/RDNA4 | V620/RDNA2 |
| --- | ---: | ---: |
| accepted package prefill tok/s, `512` prompt | 2.912 | 2.520 |
| accepted package decode tok/s, `256` generated | 0.141 | 0.139 |
| accepted package consumed VRAM GiB | 26.257 | 26.247 |
| accepted package KV cache bytes | 50331648 | 50331648 |
| materialized-AQ baseline decode tok/s | 0.140 | deferred |

T4 reference guard:

| target | layers | max MSE | max mean abs diff | max abs diff | min cosine similarity | verified |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| R9700/RDNA4 | 12 | 0.003055947561 | 0.043167665239 | 0.629638672 | 0.994777962 | true |
| V620/RDNA2 | 12 | 0.003055947561 | 0.043167665239 | 0.629638672 | 0.994777962 | true |

## SQ Format Requirements Implied by Measurements

The first sq format should make these properties explicit:

- compact resident weight bytes;
- materialized working-set bytes;
- materialization granularity, such as tensor, projection, layer, or layer window;
- per-family quantization metadata;
- tensor scale and group scale encoding;
- optional row-scale override metadata;
- codebook or equivalent reconstruction payloads;
- runtime-visible timing fields for materialization, prefill, and decode.

The first sq runtime prototype should avoid these current failure modes:

- whole-model f32 residency;
- repeated long decode measurement on a known `0.14 tok/s` path;
- conflating storage format quality with host/runtime orchestration overhead;
- treating KV cache compression as the main priority for the current `512/256` single-request case.

## Initial SQ Candidate Direction

Start from the accepted correctness policy rather than a new unvalidated quantization family:

- use the accepted `qwen35_9b_p4p46_hidden3994_v1` behavior as the correctness anchor;
- preserve row-scale override support;
- keep per-family metadata so attention, linear-attention, and MLP matrices can diverge later;
- store compact payloads resident and materialize only the active working set;
- record materialization cost separately from matmul and decode-loop cost.

The first useful performance comparison is not BF16 vs AQ. It is:

1. current materialized-AQ f32-resident path;
2. sq compact-resident path with bounded materialized working set;
3. same short reference guard and short decode probe.

## Deferred Items

- true BF16 full decoder baseline;
- tensor parallel;
- batch and continuous batching;
- server API;
- tokenizer integration;
- long `2048/512` stretch runs until decode path improves;
- final logits or generated-token reference agreement.

## Risks

| risk | impact | handling |
| --- | --- | --- |
| decode overhead is outside weight format | sq may reduce VRAM but not immediately improve tok/s | record materialization time separately and keep short decode probes |
| BF16 baseline is deferred | speed comparison may look incomplete | state that current comparison is against materialized-AQ lower bound |
| reference guard is hidden-state only | generated token correctness is not fully proven | require logits/generated-token guard before product benchmark claims |
| row-scale overrides become format complexity | format may overfit current policy | keep overrides optional and explicitly scoped |
| V620 memory is close to current f32 residency | future stretch contexts may fail | prioritize compact residency before longer contexts |
