# SQ9_0 V620 実機検証（サーマルガード付き再実行）

## 前回の要点

- 前回の V620 実測は card1（0000:43:00.0）で実行され、junction 100 C / 148 W
  に達したため、ユーザーの指示で SIGKILL を送って緊急停止した。
- V620 はパッシブ冷却で、card1 は card0 より明確にエアフロー条件が悪い。
  R9700 と DRM/HIP ordinal は対応しないため、単なる ordinal 指定は温度監視の
  根拠にならない。
- 前回の未コミットのベンチマーク、ビルド補助、部分的な結果は捨てずに引き継いだ。
  bdf43 の過去ログは履歴として残し、今回の判定値には混ぜない。

## 今回の変更点

- tools/bench-sq9-v620-viability-hip.cpp を既存実装から拡張し、選択 HIP
  デバイスの hipDeviceGetPCIBusId と DRM card*/device の PCI BDF を照合して、
  一致カードの hwmon/temp2_input（temp2_label=junction）だけを読むようにした。
- junction >= 85 C を fail-closed の中断閾値にし、各 warmup・各 timed launch の
  前後でサンプルを取り、測定点前後に <= 42 C のクールダウンを入れた。開始、
  終了、ピーク温度、サンプル数、再試行回数を JSONL に残す。
- final point も測定後に冷却するようにし、full suite には --shape と --m-values
  の明示指定を必須化した。これで無意識の all-shape sweep を拒否する。
- card0（gfx1030, 0000:03:00.0）だけを HIP_VISIBLE_DEVICES=2 で可視化して
  実行した。R9700 では測定を実行していない。
- 数値正当性を CPU 参照と照合してから M=1 を短時間で開始し、統計反復、M=8、
  M=32、M=128 へ段階的に広げた。最大 junction は有効結果で 51 C であり、
  85 C ガードは発動しなかった。
- M=512 は旧実行時に --shape が漏れ、古い既定値の all による部分 sweep が始まった。
  最大 59 C、42 C への復帰待ち 43 C で正確な PID を SIGKILL し、全 GPU 実測を
  終えた。この部分ログは保存したが、比較結果に採用していない。
- 判定は M=1 の最速 SQ9_0（LDS）が SQ8_0 比 +6.069% で、+7.29% の採算条件を
  満たさない。M>=8 には lane 版が条件を上回る観測があるが、decode の結論を
  反転させるものではない。
- SQ8_0 の dequant-only は 0.245603 ms、対応する control は 0.110122 ms であり、
  ISA でも SQ8_0 は静的 v_* 命令 377 本、lane SQ9_0 は 250 本だった。これは
  fallback dequant の大きな ALU/control 成分を支持するが、全 GEMV の純粋な
  ALU 律速を単独で証明するものではない。
- 生データ、温度履歴、ISA 分析、サマリは
  benchmarks/results/2026-07-26/sq9-v620-viability/ に記録した。

## 次の行動

- 本件では SQ9_0 の実装、candidate、campaign、release、activation は行わない。
- decode 採用を再検討するなら、まず固定クロックまたは等価な電力状態管理の下で
  M=1 を再現し、M=2–7 の crossover、実モデルの quality、KV/context を含む
  model-loop を別承認で測る。
- バッチ用途の M>=8 の条件付き優位性は、統計反復と他 projection を追加してから
  別用途として評価する。今回の partial M=512 ログは統計結果として使用しない。

