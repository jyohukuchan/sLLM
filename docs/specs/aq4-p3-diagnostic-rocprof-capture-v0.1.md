# AQ4 P3 one-case diagnostic rocprof capture v0.1

## 前回の要点

P2 one-case smokeはresident processを1回だけ起動し、2 warmup + 10 measuredを実行する。P3 producerはrun別kernel/HIP API CSV、完全なcapture capability、resident raw/summary/identityのhash bindingを要求する。既存family profilerはkernel traceだけを取得し、通常のrocprof CSVにはrun境界がないためwarmupを証明付きで除外できなかった。

## 今回の変更点

`tools/capture-aq4-p3-diagnostic-profile.py`は、marker付きone-case runnerをrocprofv3の1 subprocessで包み、kernel dispatch、HIP runtime API、memory copy、ROCTx markerを同時captureする。artifactとsplit traceは新規pathだけへ発行し、既存pathを再利用・上書きしない。

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
-- EXACT_RUNNER_COMMAND...
```

profilerとrunner executableはabsolute、ancestor symlinkなし、executable regular fileとして読み、path/inode metadata/SHA-256を固定する。command全体のcanonical SHA-256もartifactへ保存する。captureはprocess groupを分離し、timeout時はSIGINT後にSIGKILLする。nonzero、signal、OOM候補の137/-9を成功扱いしない。

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

最小実装箇所は`tools/run-aq4-p2-resident-batch.py`のmain run loopにある`_send(... command=run ...)`直前と、`validate_run(_recv(...))`直後である。明示的profiling flagがない通常runはmarker libraryをloadせず、挙動を変更しない。marker libraryはabsolute path、ancestor symlinkなし、single-link executable/DSO、inode identity、SHA-256、必要ROCTx begin/end/flush symbolを起動前後に固定する。

runner sourceを変更すると、prepared one-case bundleの`trusted-runner.py`、bundle manifest、trust roots、SHA256SUMS、launcher固定hashの再生成が必要である。これらは別作業であり、本capture toolは未更新runnerからmarkerなしのartifactを作らない。

## warmup除外とsplit

全source rowのstart/endをmarker containmentで分類する。marker境界を横断するrowは拒否する。0,1のrowは保持せず、2..11だけをrun別CSVへ書く。kernel CSVにはmarkerが証明した`Phase=prefill`を付加する。

各measured runはnon-empty kernel/HIP API traceを必要とする。kernelは保守的family mappingでunknownまたは複数family一致を拒否する。HIP APIは方向不明のmemcpyとunknown synchronizeを拒否する。memory copyは明示的D2H/H2D/D2D/H2H/peer種だけを受理する。

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

artifactはidentity、resident summary/raw、run ID、resident session、case ID/SHA、runtime device、profiler、runner executable、source 4 trace、capability、run別3 traceをSHA-256で結び、artifact自身もself-hashする。producer用bindingはmeasured index 2..11のexactly 10件で、`measurement_eligible=false`である。

rocprof overheadを含むため、このartifactから得るlatency/throughputはperformance promotionに使用しない。one-caseであり7 prompt coverageも満たさない。producerはdiagnostic rawだけを生成でき、selector promotion inputにはならない。

## 次の行動

runner marker hookとprepared bundle/trust hashを別commitで実装・再生成する。その後、明示された保守窓でだけfakeではないone-case diagnostic captureを行う。marker 12件、unknown kernel/APIなし、raw/summary binding一致を確認するまでP3 promotion evidenceへ進めない。
