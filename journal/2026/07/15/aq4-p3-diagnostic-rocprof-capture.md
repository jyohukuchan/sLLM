# AQ4 P3 diagnostic rocprof capture

## 前回の要点

- P2 one-case smokeはresident process 1、2 warmup + 10 measuredを実行する。
- 既存family profilerはkernel traceだけで、run境界とHIP API/memory domainを持たない。
- P3 producerはhash-bound capabilityとrun別kernel/API traceを要求する。

## 今回の変更点

- `capture-aq4-p3-diagnostic-profile.py`を追加した。
- rocprof 1 subprocessでkernel、HIP runtime、memory copy、markerをcaptureするcommandを固定した。
- exact 12 markerをrun/session/case/hashへ結合し、warmup 0,1を除外してmeasured 2..11をsplitする。
- resident identity/raw/summary/device、source/split traces、profiler/runner command、capabilityをhash bindingする。
- unknown kernel/API/memory、marker crossing/missing、trace再利用、output overwrite、timeout/OOMをfail-closedにした。
- producer diagnostic modeは1件またはmarker済みexact10件のprofile bindingを受理する。いずれもpromotion不可である。
- fake rocprof、marker欠落、timeout/OOM、no-reuse、10-run split、producer diagnostic buildのtestsを追加した。
- 独立QAを受け、warmup/marker外を含む全source kernel/API/memoryをsplit前に分類するよう修正した。
- timeout cleanupをprocess-group生存確認、SIGINT、SIGTERM、SIGKILL、parent wait、全子消失確認へ強化した。
- assemble失敗時は今回生成したsplit/capability/artifactだけをcleanupし、sourceを保持して再assemble可能にした。

## 検証

- GPU、service、model loadは実行していない。
- `tests/test_capture_aq4_p3_diagnostic_profile.py`: 8 passed
- capture + producer + profiler + selector: 82 passed
- py_compile: passed

## 残課題

- live runnerにはROCTx run marker hookがまだない。
- `tools/run-aq4-p2-resident-batch.py`とprepared bundle/trust hashの更新が必要である。

## 次の行動

- 別Lunaがrunner marker hookを実装し、bundle/trust chainを再生成する。
- hook独立QA後にだけ実R9700 diagnostic captureを行う。
