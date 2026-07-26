# SQ8_0 数値ゲート v0.2 — artifact-FP32 相対品質の凍結

## 結論

`SQ8_0` 最適化の数値 admission を、CK/direct との multi-step bitwise equality
だけに依存しない新しい検証計画として凍結した。主基準は、同一 `SQ8_0` artifact
を strict F32 で実行した参照に対して、candidate が matched CK/direct control と
同等以上に近いことである。

機械可読正本は
`docs/plans/sq8-numerical-gate-v0.2-relative-to-fp32-reference.json`、SHA-256 は
`64a43c032570bed8086e3c441b0774cc470c5ab1e8c67f99e02af2b6307f72bf`。

この task では GPU、候補再評価、activation、campaign、systemd、active manifest、
`/opt/ullm`、既存 evidence を変更していない。

## 設計

- 主参照は量子化前 source model ではなく、固定 artifact の FP8 payload/scales を
  F32 に正確に復元し、runtime activation quantization を通さず full model を実行する
  `artifact_fp32_strict_v1` とした。
- source model の F32 実行は artifact 化の損失も含むため、kernel path の通落ではなく
  secondary diagnostic とした。
- logits の relative L2 / P99 / max abs / mean+P99 KL、final hidden、全 layer hidden、
  top-1、top-10 を個別 gate とした。composite score による相殺は許可しない。
- candidate の連続誤差は control median の 1.05 倍、control repeat envelope、F32 由来の
  fixed floor を超えてはならない。top-1 は 95% 片側 Wilson と 64-token block bootstrap
  を併用し、non-near-margin candidate-only regression は 0 件とした。
- primary decode は candidate 非依存の 7 stream / 4096 positions。teacher forcing の token
  列は artifact-FP32 reference が先に生成して hash 化し、control/candidate が共有する。
  127/128、255/256、511/512、1023/1024、4095/4096 と M=128 prefill/tail は別途必須 coverage。

数値、入力、選択法、評価順序は候補の結果を見る前に JSON へ固定した。以後の変更は
v0.2 の後付け緩和ではなく、新 version と新規測定になる。

## FP32 参照の確認結果

既存 CPU 経路は full-model strict-FP32 reference を満たさない。

- `sq_reference.rs` は canonical artifact の projection-only reference で F64 accumulator。
- `sq_optimized_reference.rs` も dynamic activation quantization を含む projection-only F64
  path。
- `cpu_reference_executor.rs` は small F32 ModelGraph 向けで SQ8 weight materialization が
  なく、resource guard も full Qwen execution 用ではない。

よって現状の CPU で full-model artifact-FP32 が実行可能か、および総所要時間は
**未確認**である。v0.2 はこれを推測で補わず、独立 CPU executor の実装、decoder
conformance、同一条件二回の byte-identical capture、8-step CPU pilot を先行条件にした。

## 凍結順序

1. artifact 固定時の主参照意味論、被覆、全指標、non-inferiority 式、統計手順を設計した。
2. JSON を作成し、`jq` で JSON として検証した。
3. JSON の SHA-256 を計算して上記値を記録した。
4. plan/journal に hash と順序を記録した。
5. candidate/control/reference の model 実行はその後の別作業に残した。

## 次の行動

1. CPU-only artifact-FP32 full-model runner を適格化する。
2. reference token/tensor capture を作る。
3. GPU 窓で control 3 repetition を一窓、5 候補を一候補一窓ずつ 2 repetition で測る。
4. pass 候補だけ独立 confirmation を追加するため、必要 GPU 窓は最低 6、最大 11（実時間は未確認）。
