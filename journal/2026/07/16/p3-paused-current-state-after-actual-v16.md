# P3 paused current state after actual-v16

作成時刻: `2026-07-16T11:12:20+09:00`

ユーザーの指示により、P3 actual-v16後の作業をここで一時停止する。この記録の作成とcommit以外には、GPU、service、process、lock、actual、finalizer、recovery、evidence rootsを操作していない。

## 最終安全状態

- source commit前の最終read-only probe（`2026-07-16T11:05頃+09:00`）では、`ullm-openai.service` はmain PID `3707817`、worker PID `3708189`、`active/running`、`NRestarts=0` だった。
- AMD-SMI ownerとKFD ownerはworker PID `3708189`だけ、targeted processは空、service/worker/GPU/hash/formal healthはmaintenance restore poststateと一致した。
- これは停止時点の最後の観測であり、再開時には同じread-only phase-aware probeで必ず再確認する。現在も同じだとは観測なしに断定しない。

## 完了済みauthority chain

- capture source: `418e507214b2a4c0352ac8867bf9689b81948ca4`
- capture/launcher lock isolation tests: `376b733b097db37701529014e4e698093976d689`
- launcher-v12: `780a68007d424e1cf3f53d4e60728161ce6d13d4`
- execute binding-v12: `9fdab4c5aa2c60813fbe9c0527ac0bdffa725044`
- maintenance closure cascade: `fd0b964d8467cd34ad7f8a012ee1f91869a71560`
- ready-v18: `42856dbf80ca06b51a70994b224151320b0011ef`
- offline reassembly-v13: `f1f92ad90834514f93ec92690f0285ea2b515c63`
- operator-v16 authority integration: `4869fde48ca872da70b09b029ebdd9da169fc4b1`
- quiet-window-v21: `e1ad3423ae19f16c0bfd7f4648f54e6c81d91031`
- command-v16: `7ec8189d389b81f5b7d77e050707069c11dd6ae1`
- previous sealed actual-v15: `99faf0066b93eb021fa83bea1b1a0193d9a79fd4`
- failed-finalization recovery source: `0302979a87009655bcae4850b314e97767024670`

## actual-v16とfinalizer境界

- actual-v16は外部実行観測上、exact-one invocation `1/1`、return code `1`、retry `0`で終了した。再実行は禁止する。
- `finalize-actual`は1回呼ばれ、`lock/residual cleanup differs`で失敗した。再呼出は禁止する。
- workspaceのartifactだけからは、outer wait return code、canonical start/end、exact invocation count、retry観測、finalizer invocation count、finalizer error streamを独立再導出できない。これらを後から推定値で補完または捏造しない。
- workspaceから直接確認できるのは、command-v16のmaximum invocation 1/retry forbidden、unique actual outputs、maintenanceとcaptureの内部exact-one counters、failed statusである。

## current partial roots

| root | 現在状態 | 内容 |
|---|---|---|
| `resident-one-case-smoke-profile-maintenance-evidence-v13` | sealed `0555` | 5 files、全member `0444`、`SHA256SUMS`検証済み、status failed、restore passed |
| `resident-one-case-smoke-profile-execute-v12` | unsealed `0775` | 6 files、`SHA256SUMS`なし、workload summary complete |
| `resident-one-case-smoke-profile-execute-evidence-v12` | sealed `0555` | 6 files、全member `0444`、`SHA256SUMS`検証済み、launcher status failed |
| `aq4-p3-diagnostic-rocprof-capture-v12` | unsealed `0700` | 39 filesと`measured-runs/` 1 directory、`SHA256SUMS`なし、capture artifact complete_diagnostic |
| `resident-one-case-smoke-profile-operator-result-v16` | partial unsealed `0755` | raw stdout/stderr 2 filesだけ、result JSON/SUMSなし |
| `resident-one-case-smoke-profile-actual-audit-v16` | absent | 未生成 |

## failure root causeと保存済みworkload

- `rocprof.stderr`は558 bytes、SHA-256 `5009fb9c86decf8c0e5f857923cfaa6753f548c1b9187a29d23374ef9381561a`である。
- 内容はROCprofの`output_stream.cpp:111] Opened result file:`という情報メッセージ2行だけで、marker API traceとagent info CSVの出力先を通知している。runner failure stderrではない。
- maintenanceはこのprofiler stderrを`profiled runner stderr`としてzero-stderr条件で拒否した。launcher diagnosticsが正式化されず、maintenanceの昇格lifecycleが`unknown`になり、trusted lock substrate cleanupがfail-closeで未実施になった。その結果、finalizerが`lock/residual cleanup differs`で停止した。
- raw lifecycleはmaintenanceとlauncherで完全一致し、`rocprof_started=true`、`runner_completed=true`、`children_remaining=[]`、`cleanup_passed=true`だった。
- workloadはresident model load 1、warmup 2、measured 10、transaction 12、driver cleanup passedで完了した。
- capture artifactはschema v2、`complete_diagnostic`、self-hash valid、measurement/promotion eligible falseである。failure evidenceとして保存し、成功測定へ昇格しない。

## retained lock substrate

- `/run/ullm/r9700.lock`はmaintenance生成時と同じdevice `26`、inode `830148`だった。
- 最終read-only確認ではkernel FLOCK WRITE holderはservice main PID `3707817`だけだった。復旧serviceが同じinodeを取得しているため、削除、unlink、rmdir、lock cleanupの再試行は禁止する。
- lock file payloadは旧actual driver PID `3705490`とrun idを残すstale recordであり、current owner authorityではない。current ownerはkernel lock tableで判定する。

## recoveryと未完作業

- commit `0302979a87009655bcae4850b314e97767024670`に専用`recover-failed-finalization`とstatic validatorを追加した。targeted testsは20件passした。
- recovery sourceの独立QAは未完で、専用subcommandは未実行である。`finalize-actual`の再実行で代替してはいけない。
- maintenanceのbenign ROCprof stderr classifier preworkは別担当の未commit変更で、targeted testsは190件passと報告されている。現在のdirty source/testをauthorityとして扱わない。
- launcher-v13、operator-v17、次namespace/cascadeは未着手または未確定である。actual-v16 rootsを次のexecution authorityとして再利用しない。

## journal作成時のworktree

HEADは`0302979a87009655bcae4850b314e97767024670`、treeは`ddaf275374e5c8f190bd7281ec129a2437286c86`だった。このpause journalはHEAD後に作成するため、commit後はHEADが1つ進む。authority chainの実装commit自体は上記のまま変わらない。

`git status --short`のjournal作成直前の正確な出力は次のとおりだった。

```text
 M crates/ullm-engine/src/bin/ullm-aq4-fidelity-capture.rs
 M crates/ullm-engine/src/bin/ullm-aq4-p2-full-model.rs
 M crates/ullm-engine/src/bin/ullm-aq4-p2-path-oracle.rs
 M tests/test_aq4_p2_resident_smoke_maintenance.py
 M tools/run-aq4-p2-resident-smoke-maintenance.py
?? benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p0/
?? benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p1/
?? benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/fidelity-calibration-capture-v0.1/attempts/source-attempt-20260714T164526Z/
?? benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/fidelity-calibration-capture-v0.1/input/
?? benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-profile-execute-evidence-v12/
?? benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-profile-execute-v12/
?? benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-profile-maintenance-evidence-v13/
?? benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-profile-operator-result-v16/
?? benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p3/aq4-p3-diagnostic-rocprof-capture-v12/
?? docs/reference/
```

- 変更中のmaintenance source/testはbenign classifier担当の未commit作業で、このjournal担当の所有物ではない。
- Rust 3 files、P0/P1、fidelity capture、`docs/reference/`は既存または別担当の変更で、このjournal担当の所有物ではない。
- actual-v16の5 untracked rootsはpartial evidenceであり、削除、変更、stage、restoreしてはいけない。

## 再開手順と禁止事項

1. 最初にHEAD、worktree、6 current rootsのmode/member/hash/SUMS/self-hashをread-onlyで再確認する。
2. service/worker epoch、AMD-SMI/KFD owners、targeted process、formal health/hash、kernel FLOCK holderとinodeをread-onlyで再確認する。
3. 外部SQ8 workloadへ干渉せず、kill、restart、service stop/start、GPU execution、lock削除を行わない。
4. workspaceから証明できないrc/start/end/finalizer metadataを捏造しない。
5. recovery commit `0302979a...`を独立QAし、source authorityと全preconditionを確認する。
6. QA通過後にだけ専用`recover-failed-finalization`を1回実行する。actual、maintenance、`finalize-actual`は再実行しない。
7. 6 rootsのseal、result/audit schema/self-hash/SHA256SUMS、static validator、Git blob/archive一致を独立監査する。
8. failure seal完了後にだけ、benign classifierの正式authority化、launcher-v13/ready、operator-v17、次namespaceのcascadeへ進む。

## 前回の要点

- actual-v16はworkload/captureを完了したが、benign ROCprof stderrの誤分類からfail-close cascadeになり、finalizerも1回失敗した。
- recovery-only sourceはcommit済みだが未QA・未実行で、partial evidenceは未封印である。

## 今回の変更点

- ユーザー指示による停止時点のauthority、live safety、partial roots、root cause、dirty worktree、再開条件を1つのjournalへ固定した。
- 実行状態、evidence、既存dirty pathsは変更していない。

## 次の行動

- 再開までは何も実行しない。
- 再開時はread-only safety再確認から始め、独立QA後の専用recovery一回だけを次のmutation候補とする。
