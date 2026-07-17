# SQ FP8 runtime loader smoke

## 前回の要点

- T0はR9700/AQ4/SQ候補ID/schemaのfreezeまで完了した。
- T1はSQ比較用JSONLにcandidate、artifact、prefill executor、KV cache bytesなどを残す入口まで進んだ。
- T2はFP8 SQ artifact writerとmanifest仕様まで進んだが、runtime load pathとshort prompt guardが残っていた。

## 今回の変更点

- `crates/ullm-engine/src/sq.rs` を追加し、`sq-fp8-artifact-v0.1` のmanifest読込、検証、FP8 tensor選択、FP8 E4M3 + F32 scaleの行単位materializeを実装した。
- `ullm-engine sq-fp8-materialize-smoke` を追加し、選択行をF32化してruntime bufferへコピーし、読み戻し一致を確認できるようにした。
- 小さい4x8 fixture artifactを作り、CPU device `0` とR9700 device `2` の両方で `roundtrip_max_abs_diff=0` を確認した。
- 計画文書とstate freezeを、T2 runtime load pathがpartial doneになった状態へ更新した。

## 次の行動

1. SQ FP8 materialize helperを既存package model load pathへ接続する。
2. selected FP8 tensorsだけをSQ artifactから、残りをAQ4/package側から読む混在load方針を決める。
3. short prompt guardでAQ4 baselineとSQ FP8 candidateの出力品質を比較する。
4. guardが通ったらT3のbatch/cold/cached-prefix/decode gridへ進む。
