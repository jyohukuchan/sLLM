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
- interoperability QAとしてprofilerのabsolute invocation、symlink chain、resolved inode、single-link、expected SHA-256を必須化した。profilerはread-only FDを保持し、初回・spawn直前・終了後に再検証して同じ`/proc/self/fd`から実行する。
- target commandはself-hash付きexact argv manifestへ変更した。argv[0]を含む入力file hashと出力pathをargument indexへ結び、absolute argvを漏れなく分類し、spawn直前と終了後に再検証する。
- 実行所有順を`outer maintenance harness -> capture tool -> rocprof child(new process group) -> launcher -> runner`へ固定した。timeout cleanupはrocprof child groupだけへSIGINT、SIGTERM、SIGKILLを段階送信し、group全消失を要求する。
- timeout/nonzero/launch失敗ではread-only `capture-failure.json`へnon-promotion、cleanup状態、outer harness非signal、stdout/stderr hash、固定入力contextをself-hash付きで残す。
- 独立QA修正として`--target-command-manifest-sha256`を必須化し、manifest file全体の期待SHAとmanifest内self-hashを別々に検証する。manifestはread-only同一FDでinitial、spawn直前、終了後にpath/inode/file SHA/self-hashを固定する。
- post-spawn verifierの失敗を`run_profile`のfailure経路内へ移し、path swapを含む全CaptureErrorでatomic read-only failure evidenceを発行してからraiseする。failure evidenceがあるoutputからsuccess artifactを発行せず、両者を併存させない。

## 検証

- GPU、service、model loadは実行していない。
- `tests/test_capture_aq4_p3_diagnostic_profile.py`: 11 passed
- fake outer harness sentinelがcapture timeout後も生存し、子group消失後にrestoreを完了することを確認した。
- profiler SHA不一致・invocation symlink差し替え、target absolute argv未分類を拒否することを確認した。
- manifest argvを並べ替えてself-hashを再計算しても期待file SHA不一致で拒否し、post-spawn path swapがfailure evidenceを残してsuccess artifactを禁止することを確認した。
- capture + producer + profiler + selector: 85 passed
- py_compile: passed

## 残課題

- runner sourceのROCTx run marker hookは実装済みだが、prepared bundle/trust hashには未反映である。
- outer maintenance harnessからcapture tool、profiler、target launcher、target manifestを4段階で固定する再束縛は完了した。

## 次の行動

- 別の明示承認がある場合にだけ、再生成済みprofile-ready artifactから実R9700 diagnostic captureを行う。
