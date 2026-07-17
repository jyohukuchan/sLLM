# SQ T0-T2 plan update

## 前回の要点

- cached-prefix attentionは `cached_prefix_rdna4_fp8_auto` まで進み、SQ候補評価を始める前提は整った。
- 次の主タスクはattention kernel追加ではなく、`sq-fp8-w8a16-r9700-v0` のartifact、result schema、batch throughput記録へ移った。

## 今回の変更点

- `docs/plans/fp8-sq-r9700-batch-throughput-prefill-plan-v0.1.md` にT0-T2の現在地、完了条件、次の行動を追記した。
- T0のfreezeとして `benchmarks/results/2026-07-08/sq-r9700-state-freeze-v0.1.*` を追加した。
- T1ではbatch throughput JSONLに、SQ candidate、artifact、prefill executor、cached-prefix token数、推定attention work、KV cache bytesを残す入口を追加した。
- T2では `sq-fp8-w8a16-r9700-v0` のmanifest仕様とFP8 E4M3 payload writerを追加した。
- FP8 writerは小さいsafetensors fixtureでpayload生成を確認し、実モデルはmetadata-onlyでmanifest生成を確認した。

## 次の行動

1. Runtime側に `sq_manifest.json` を読む入口を追加する。
2. 選択tensorだけFP8 payload + F32 scaleからmaterializeする最小load pathを作る。
3. short prompt guardでAQ4 baselineと出力品質を比較する。
4. Runtime loadが通った後、T3として `batch=1/4/8` と cold/cached-prefix/decode の代表gridを保存する。
