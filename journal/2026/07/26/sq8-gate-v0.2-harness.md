# SQ8 numerical gate v0.2 evaluation harness

Date: 2026-07-26

## 前回の要点

- frozen v0.2 JSON の SHA-256 は
  `64a43c032570bed8086e3c441b0774cc470c5ab1e8c67f99e02af2b6307f72bf` であり、
  primary 4,096 decode、boundary、M=128 prefill、control 3/candidate 2 repetition が
  admission の前提である。
- strict artifact-F32 CPU reference は
  `benchmarks/results/2026-07-26/sq8-fp32-reference/cpu-f32-parallel-reference-v1/` で
  生成中である。今回その root に書込みはしていない。

## 今回の変更点

- consumer evaluator `tools/evaluate-sq8-gate-v0.2.py` を追加した。実行時に frozen JSON の
  raw SHA-256/schema を検証し、reference index、control 3、candidate 2 の immutable F32LE
  payload の shape/byte count/hash/finite 性を再検証する。logits relative L2/P99/max-abs/KL、
  final hidden、40 layer hidden と final norm、top-1/top-10 Wilson、64x64 block bootstrap を
  F64 で個別に再計算する。frozen corpus から 4,210 capture position と 626 layer-required
  position を再導出し、set equality を要求する。
- repeat envelope は frozen JSON の説明に従い、upper を `max(control)-median(control)`、lower を
  `median(control)-min(control)` とした。P99 interpolation、bootstrap PRNG、同率時の control
  選択は JSON に規定がないため、receipt に nearest-rank / SHA-256+PCG64 / lower index として
  明示した。基準 JSON の数値はコードに複製していない。
- `ullm-sq8-gate-capture` と plan launcher を追加した。private teacher-forced API が CPU reference
  token stream だけを次 input に使い、必要 position の logits/final/layer trace を F32LE で保存する。
  source-tile 128/256 は exact `ULLM_EXPERIMENTAL_SQ8_PAGED_DECODE_SPLIT_ALLOW_MULTITILE=1` の
  child-process opt-in 時だけ direct containment fallback を迂回する。unset/default は従来通り
  direct dispatch である。plan は incomplete reference index を `blocked_reference_or_capture` として
  GPU 起動前に拒否し、capture manifest は executable/git/feature/device/selector/HIP guard/plan hash を
  記録する。consumer は selector 以外の runtime configuration が control/candidate 間で同一か確認する。
- Flash2、tile128、tile256 は full v0.2 route を持つ。handwritten WMMA は private M=1-only
  selector のため required M=128 が欠けて blocked/non-qualifying、`SQ8_1` W8A8 は frozen scope
  `SQ8_0` 外として別 quality gate に残す。どちらも v0.2 pass として偽装しない。

## 検証

- active reference から read-only partial index（179 position）を作った。reference を control 3 と
  candidate 2 に alias した self-consistency は、測定済み logits/final/40 layer/final norm の全 gate
  で failure 0 だった。coverage は primary 144/4,096、stream 1/7、block 2/64、全 required
  position 179/4,210 なので status は `test_only_harness_verification` であり、admission result では
  ない。最新 receipt は
  `benchmarks/results/2026-07-26/sq8-gate-v0.2-harness/self-consistency-final-index/result.json`。
- 同じ partial input の最初の logit を `+100` した candidate は prefill checkpoint の aggregate/P99
  relative L2、max-abs、mean/P99 KL、top-1 Wilson、hard top-1 regression で失格した。position は
  `m128_chunks_with_declared_tail:chat-p2048-g512:prompt:00127` と receipt に記録された。receipt は
  `benchmarks/results/2026-07-26/sq8-gate-v0.2-harness/intentional-failure-final-index/result.json`。
- partial index に対する Flash2 plan preparation は、4,063 required position不足を示す
  `blocked_reference_or_capture` receipt で停止した。GPU subprocess は起動していない。
- Python unit test 8 件、default Rust type check、`rocm-ck-gfx1201` と
  `rocm-handwritten-projection-gfx1201` の capture binary type check を通した。
  GPU execution、systemd、active manifest、`/opt/ullm`、activation/campaign は実施していない。

## GPU 実行計画

- standard build の control triplet は Flash2/tile128/tile256 で共有できるので、admission 最小は
  control 1 + candidate 3 の 4 isolation window である。pass candidate が `P` 件なら independent
  confirmation は新 control 1 + candidate `P` で `+1+P` window。handwritten diagnostic は optional
  `+1`、SQ8_1 は 0 である。
- full standard admission は 9 repetition / raw payload 約 26.47 GiB、confirmation は control 3 +
  candidate `2P` repetition / 約 `8.82 + 5.88P` GiB である。layer readback と filesystem write を
  含む full GPU runtime は未測定である。reference 完成前に GPU window を消費しないため、既存 serving
  timing から総時間を推測しない。最初の standard control capture が manifest に total/mode
  `elapsed_seconds` を書くので、それを同一 window の残り capture reservation に使う。
