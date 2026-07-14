# AQ4 P3 one-case diagnostic rocprof capture v0.1

## 前回の要点

P2 one-case smokeはresident processを1回だけ起動し、2 warmup + 10 measuredを実行する。P3 producerはrun別kernel/HIP API CSV、完全なcapture capability、resident raw/summary/identityのhash bindingを要求する。既存family profilerはkernel traceだけを取得し、通常のrocprof CSVにはrun境界がないためwarmupを証明付きで除外できなかった。

## 今回の変更点

`tools/capture-aq4-p3-diagnostic-profile.py`は、marker付きone-case launcher/runnerをrocprofv3の1 capture subprocessで包み、kernel dispatch、HIP runtime API、memory copy、ROCTx markerを同時captureする。artifactとsplit traceは新規pathだけへ発行し、既存pathを再利用・上書きしない。

安全な所有順序は`outer maintenance harness -> capture tool -> rocprof child -> launcher -> runner`である。service停止・復帰を所有するouter maintenance harnessはcapture toolの外側に置く。capture toolはrocprof childだけを`start_new_session`で新しいprocess groupへ分離し、outer harness自身とその復帰処理へsignalを送らない。

実行設定は次である。

```text
rocprofv3
--kernel-trace
--hip-runtime-trace
--memory-copy-trace
--marker-trace
--output-format csv
--output-directory ABSOLUTE_NEW_DIRECTORY
--output-file SAFE_NAME
-- EXACT_HASH_BOUND_TARGET_COMMAND...
```

CLIは`--profiler-path`、`--profiler-sha256`、`--target-command-manifest`を必須にする。profiler pathはabsoluteかつ`..`なしで、invocationのsymlink chain、resolved path、single-link executable regular file、inode identity、SHA-256を固定する。resolved profilerをread-only FDで開いたまま、初回検証、spawn直前検証、終了後検証を行い、実行自体も同じFDの`/proc/self/fd/<n>`を使う。path再openだけに依存しない。

target command manifestは`ullm.aq4_p3_profile_target_command.v1`、`status=bound`、self-hash、exact `argv`、`input_files`、`output_paths`を持つ。`input_files`はargument index、absolute path、SHA-256、executable flagを結び、argv[0]をsingle-link executable regular fileとして必ず固定する。`output_paths`もargument indexとabsolute pathを結ぶ。argv内のabsolute pathは入力または出力へ漏れなく一度だけ分類し、入力manifestとファイルidentityをspawn直前と終了後に再検証する。capture時の出力pathは未作成でなければならず、正常終了後に生成済みであることを要求する。command全体のcanonical SHA-256もartifactへ保存する。

timeout時はrocprof child groupへSIGINTを送り、残存時だけSIGTERM、SIGKILLの順で段階終了する。親をwaitし、process group全体の消失を確認できなければcleanup失敗として拒否する。outer maintenance harnessのprocess groupは対象にしない。nonzero、signal、OOM候補の137/-9を成功扱いしない。失敗時は`capture-failure.json`をread-onlyかつself-hash付きで発行し、non-promotion、cleanup完了可否、outer harnessへsignalを送っていないこと、stdout/stderr hash、固定入力contextを残す。

## marker hook契約

同じrocprof sessionにexactly 12個のbalanced rangeが必要である。clock domainは`rocprofv3_monotonic_ns`で、start/endは`0 <= start < end`のint nanoseconds、rangeはindex順で重なってはならない。

marker名は次のexact schemaである。

```text
ullm.aq4_p2.run.v1/
run_id=<resident batch run id>/
session_id=<resident session id>/
case_id=<case id>/
case_sha256=<64 lower hex>/
run_index=<0..11>/
run_kind=<warmup|measured>
```

実際は改行せず1文字列に連結する。index 0,1は`warmup`、2..11は`measured`である。beginはrunnerが`command=run`をresident stdinへ送る直前、endは対応する`run_complete`を受信してexact schema・status・index・kindを検証した直後である。exception、OOM、timeoutでもfinallyでrangeを閉じ、flush完了後に次rangeを開始する。missing、duplicate、unknown field、unbalanced、crossing、clock逆行、rawとのrun/session/case/hash不一致を拒否する。

実装箇所は`tools/run-aq4-p2-resident-batch.py`の`execute_resident_run`で、main run loopから呼ぶ。`_send(... command=run ...)`直前にpushし、`validate_run(_recv(...))`とindex/kind検証直後にpopする。明示的`--profile-roctx-ranges`がない通常runはmarker libraryをloadせず、挙動を変更しない。marker libraryはinvocation symlink chain、resolved実体、inode identity、SHA-256、`roctxRangePushA`/`roctxRangePop` symbolを起動前後に固定する。

runner sourceは実装済みだが、prepared one-case bundleの`trusted-runner.py`、bundle manifest、trust roots、SHA256SUMS、B sidecar、launcher/harness固定hashの再生成が必要である。これらは別作業であり、本capture toolは未更新runnerからmarkerなしのartifactを作らない。

## warmup除外とsplit

全source rowのstart/endをmarker containmentで分類する。marker境界を横断するrowは拒否する。0,1のrowは保持せず、2..11だけをrun別CSVへ書く。kernel CSVにはmarkerが証明した`Phase=prefill`を付加する。

各measured runはnon-empty kernel/HIP API traceを必要とする。分類検証はsplit後のmeasured rowだけでなく、warmupとmarker外を含む全source rowへ先に適用する。kernelは保守的family mappingでunknownまたは複数family一致を拒否する。HIP APIは方向不明のmemcpyとunknown synchronizeを拒否する。memory copyは明示的D2H/H2D/D2D/H2H/peer種だけを受理する。集計とproducer bindingだけをmeasured 2..11に限定し、warmup内のunknownを無視しない。

assemble中にunknown、marker不整合、hash不一致などが判明した場合は、今回生成した`measured-runs/`、`capture-capabilities.json`、未発行artifactをcleanupする。source traceを保持した同じprofile output directoryへ、原因を修正した後に`assemble`を再実行できる。既存のsplit/capability/artifactが開始時から存在する場合は上書きも削除もせず拒否する。artifactとcapabilityはtemporary file、fsync、atomic renameで発行する。

## capabilityとhash binding

capability schemaは`ullm.aq4_p3_rocprof_capture_capabilities.v1`で、file SHAとself-hashを持つ。次をすべてtrueにする。

```text
domains:
  kernel_dispatch
  hip_api
  memory_copy
  d2h_memcpy
  stream_synchronize
  device_synchronize
rocprof_config:
  kernel_trace
  hip_api_trace
  memory_copy_trace
  marker_trace
  api_filter = all_functions
```

artifactはidentity、resident summary/raw、run ID、resident session、case ID/SHA、runtime device、profiler、target command manifest、source 4 trace、capability、run別3 traceをSHA-256で結び、artifact自身もself-hashする。producer用bindingはmeasured index 2..11のexactly 10件で、`measurement_eligible=false`である。

rocprof overheadを含むため、このartifactから得るlatency/throughputはperformance promotionに使用しない。one-caseであり7 prompt coverageも満たさない。producerはdiagnostic rawだけを生成でき、selector promotion inputにはならない。

## 次の行動

runner marker hookを含むprepared bundle/trust hashを再生成し、outer maintenance harnessがcapture tool自身のSHA-256とtarget command manifestを固定する。その後、別の明示承認がある保守窓でだけfakeではないone-case diagnostic captureを行う。marker 12件、unknown kernel/APIなし、raw/summary binding一致、outer restore完了を確認するまでP3 promotion evidenceへ進めない。
