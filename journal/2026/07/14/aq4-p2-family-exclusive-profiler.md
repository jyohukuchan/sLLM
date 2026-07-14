# AQ4 P2 family-exclusive profiler preparation

## 前回の要点

resident one-case inputと2 warmup + 10 measured planは準備済みだが、family別GPU時間をoverlapなしで帰属するtoolchainはなかった。

## 今回の変更点

- WRX80の`rocprofv3` 1.1.0 / ROCm 7.2.1とkernel trace CSV timestamp schemaをread-only確認した。legacy `rocprof` 2.0は存在するがdeprecatedのため新規runには使わない。
- resident commandをrocprofv3 subprocess 1回で包むwrapperと、v3/legacy CSVを読むoffline parserを追加した。
- case、identity、M、device、binary、package、policy、profiler version/command/traceをartifactへhash-bindした。
- interval unionをGPU totalとし、family exclusive/non-overlap、cross-family overlap、unclassifiedを保存する。inclusive単純合算はGPU totalへ使わない。
- prefill/decode phaseを分離し、phase markerなし、unknown threshold超過、profile overheadを含むartifactはmeasurementへ昇格させない。

## 次の行動

この作業ではGPU、resident process、model load、live serviceを実行しない。synthetic traceの正負例と既存CPU回帰を通した後、sanctioned GPU runは別作業として実施する。
