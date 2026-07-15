# P2 active fidelity attempt 7: schema recovery and policy No-Go

## 前回の要点

attempt 7のGPU capture outputはimmutable archiveへ保存済みであり、元archive、production service、GPUは変更しない。capture直後のGate post-checkは、実schemaのnested `cases.row_count` / `runtime.run.row_count`を参照せず失敗し、metrics adapterはtargetのparent bindingを受理できず停止した。

## 今回の変更点

target validatorを実schemaへ合わせ、target固有の厳密なlimits、nested row count、direct source-calibration parent binding、f32 little-endian bit相当のtop-k比較をfail-closedで検証可能にした。active artifactのfull validatorは24行、nonfinite 0、target schemaで`valid`となった。adapterは元archiveを読み取り、新規 recovery pathへmetricsを生成した。

新規証跡 `recovery/active-attempt7-schema-recovery-v0.1/` にsource/active/metrics SHA、active validator成功結果、metrics validatorの失敗結果、全19件の`logits_relative_l2 > 1.0`（最大1.2494246455220739）と他policy指標を固定した。凍結policyのrelative-L2 ceiling 1.0を緩和せず、metrics validatorはexit 1でNo-Goとなった。holdoutは未実行であり、今後も実行禁止である。

## 次の行動

calibrationはNo-Goのまま保持する。holdout/GPU/serviceの再実行は行わず、必要ならschema/実装レビューだけを別変更として扱う。
