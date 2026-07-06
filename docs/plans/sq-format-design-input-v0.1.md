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
- その後のruntime改善で、AQ4 prototypeはcontrolled v0.3 prompt suite上でR9700/RDNA4 mean decode `19.796 tok/s`、V620/RDNA2 mean decode `15.434 tok/s` に到達した。
- SQ候補比較の標準gateとして、v0.3 prompt suiteとguard bundleを追加した。

## 次の行動

1. sq format v0.1の設計では、compact resident storageとbounded materialized working setを最優先にする。
2. runtime計測では、controlled v0.3 suite、長いprefill timing probe、短いlogits guardを分ける。
3. sq候補のrun recordには、resident bytes、materialized bytes、materialization time、prefill TPS、decode TPS、output-health summary、guard bundle resultを必ず入れる。

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

Controlled prompt suite:

```text
benchmarks/prompts/pre-sq-runtime-prompt-suite-v0.3.json
```

Guard bundle driver:

```text
tools/run-package-prompt-guard-bundle.py
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

Current AQ4 prototype controlled prompt suite:

| condition | R9700/RDNA4 | V620/RDNA2 |
| --- | ---: | ---: |
| v0.3 mean decode tok/s | 19.796 | 15.434 |
| v0.3 min decode tok/s | 18.500 | 14.553 |
| v0.3 max decode tok/s | 20.166 | 15.668 |
| v0.3 mean prefill tok/s | 19.850 | 16.503 |
| v0.3 output ok / warn / not evaluated | 6 / 0 / 1 | 6 / 0 / 1 |

Current R9700/V620 guard bundle:

| check | result |
| --- | --- |
| prompt suite compared cases | 7 |
| generated token matches | 7 |
| top logits matches | 7 |
| max prefill top-logit abs diff | 0.0 |
| max decode last top-logit abs diff | 0.0 |
| standalone short-QA top logits max abs diff | 0.0 |
| bundle passed | true |

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
- using a faster path that passes TPS but fails the v0.3 output-health or token/logits guard bundle.

## Initial SQ Candidate Direction

Start from the accepted correctness policy rather than a new unvalidated quantization family:

- use the accepted `qwen35_9b_p4p46_hidden3994_v1` behavior as the correctness anchor;
- preserve row-scale override support;
- keep per-family metadata so attention, linear-attention, and MLP matrices can diverge later;
- store compact payloads resident and materialize only the active working set;
- record materialization cost separately from matmul and decode-loop cost.

The first useful performance comparison is not BF16 vs AQ. It is:

1. current materialized-AQ f32-resident path;
2. current AQ4 prototype path with v0.3 prompt suite and guard bundle;
3. sq compact-resident path with bounded materialized working set;
4. same v0.3 prompt suite, guard bundle, and short reference guard.

## SQ Candidate Acceptance Gate

An sq candidate is not comparable to the current AQ4 prototype unless all of these are recorded:

- package or runtime artifact path;
- target GPUs and device indices;
- compact resident bytes;
- materialized working-set bytes;
- materialization granularity;
- materialization wall time;
- v0.3 prompt suite summary for R9700/RDNA4;
- v0.3 prompt suite summary for V620/RDNA2 when the candidate is expected to support RDNA2;
- guard bundle summary comparing the candidate against the accepted AQ4 baseline or the intended reference;
- short golden prefix guard result;
- any row-scale override policy changes.

Minimum pass criteria for a first sq candidate:

| gate | minimum |
| --- | --- |
| correctness | short golden prefix verified on target GPU |
| output guard | v0.3 guard bundle passed |
| output health | no warnings on quality-scored v0.3 cases unless explicitly justified |
| R9700 decode | no severe regression from AQ4 v0.3 baseline without a memory-residency tradeoff |
| V620 decode | must run if RDNA2 support is claimed |
| memory | compact residency and bounded working set must be reported even if TPS is unchanged |

The first sq candidate may be accepted as useful even if decode TPS is similar to AQ4, provided it
substantially reduces resident or materialized working-set bytes and passes the same guard bundle.

## Deferred Items

- true BF16 full decoder baseline;
- tensor parallel;
- batch and continuous batching;
- server API;
- tokenizer integration;
- long `2048/512` stretch runs until decode path improves;
- independent CPU/external final-logits reference.

## Risks

| risk | impact | handling |
| --- | --- | --- |
| decode overhead is outside weight format | sq may reduce VRAM but not immediately improve tok/s | record materialization time separately and keep short decode probes |
| BF16 baseline is deferred | speed comparison may look incomplete | state that current comparison is against materialized-AQ lower bound |
| reference guard is not independent final logits | generated token correctness is stronger but still cross-device rather than external | keep v0.3 guard bundle mandatory and add CPU/external logits guard later |
| row-scale overrides become format complexity | format may overfit current policy | keep overrides optional and explicitly scoped |
| V620 memory is close to current f32 residency | future stretch contexts may fail | prioritize compact residency before longer contexts |
