# AQ4 P2 family-exclusive profiler preparation

## 前回の要点

resident one-case inputと2 warmup + 10 measured planは準備済みだが、family別GPU時間をoverlapなしで帰属するtoolchainはなかった。

## 今回の変更点

- WRX80の`rocprofv3` 1.1.0 / ROCm 7.2.1とkernel trace CSV timestamp schemaをread-only確認した。legacy `rocprof` 2.0は存在するがdeprecatedのため新規runには使わない。
- resident commandをrocprofv3 subprocess 1回で包むwrapperと、v3/legacy CSVを読むoffline parserを追加した。
- case、identity、M、device、binary、package、policy、profiler version/command/traceをartifactへhash-bindした。
- interval unionをGPU totalとし、family exclusive/non-overlap、cross-family overlap、unclassifiedを保存する。inclusive単純合算はGPU totalへ使わない。
- prefill/decode phaseを分離し、phase markerなし、unknown threshold超過、profile overheadを含むartifactはmeasurementへ昇格させない。
- resident commandを、detached resident binary、hash-bound served-model manifest、device index 1、identity-bound build commitのexact 7要素へ限定した。同一byte列の別binary、引数の入れ替え、追加commandを拒否する。
- resident/package/case/identity/policy/served-model、そこから導出するworker/package、traceをabsolute canonical path、ancestor symlink不在、inode identity、SHA-256で固定し、artifact書き出し直前に再検証する。
- profiler executableはsymlinkを含まない`/opt/rocm-7.2.1/bin/rocprofv3`へ固定し、single-link regular executable、inode identity、SHA-256をversion取得前後と最終段階で再検証する。read-only確認時のSHA-256は`13060810d6b80653631b14f0f5e33ea160c2b79a6a3a4c6850142010b48b8ec8`だった。
- symlink path、version query中のexecutable交換、別resident command、引数交換、trace/resident/package/case/policy/served-modelのTOCTOUを負例に追加した。
- synthetic test 27件とresident runnerを合わせた36件を通し、現在のprepared bundle、active product package、Cargo hard-link workerを用いたread-only bindingとprofiler version再検証が通ることを確認した。
- path/oracle回帰は19件と15 subtests、P2 prepare/evidence回帰は44件と25 subtestsが通った。Python compileと差分whitespace検査も通った。

## 次の行動

この作業ではGPU、resident process、model load、live serviceを実行しない。追補のsynthetic trace正負例、runner、current-clean CPU回帰を通した後、sanctioned GPU runは別作業として実施する。
