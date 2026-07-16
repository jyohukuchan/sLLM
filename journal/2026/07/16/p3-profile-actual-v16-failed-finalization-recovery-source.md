# P3 profile actual-v16 failed-finalization recovery source

## 前回の要点

- actual-v16 は command-v16 の一回限定枠を消費し、operator raw stdout は `status=failed`、stderr は空になった。
- workload と capture artifact は完了し、raw lifecycle は `runner_completed=true`、`children_remaining=[]`、`cleanup_passed=true` だった。
- ROCprof 自身の `Opened result file` 情報メッセージを maintenance が runner stderr と誤分類したため、formal validation が失敗した。maintenance は lifecycle を fail-close で `unknown` にし、lock substrate を削除せずserviceを復旧した。
- 従来の `finalize-actual` は `lock/residual cleanup differs` で一回失敗した。result-v16 はraw 2 filesだけ、audit-v16は不在、runtime-v12とcapture-v12は未封印のままである。

## 今回の変更点

- 既存 finalizer を再呼出しない専用 `recover-failed-finalization` と、live poststateに依存しない `validate-failed-finalization-recovery` を追加した。
- recovery は command-v16、sealed maintenance-v13、sealed execute-evidence-v12、unsealed runtime/capture、partial result raw streams、audit不在を厳密に確認する。raw lifecycleのmaintenance/launcher完全一致、capture artifact self-hash、artifact内path/hashと35 capture filesの実bytes、workload完了、driver cleanup、ROCprof情報stderrの限定形式も検証する。artifactが直接hash-bindしないagent infoとrocprof streamsは明示し、後者はempty/限定semanticとして別検証する。
- workspaceから独立に再構成できない outer wait rc、canonical start/end、exact invocation count、retry観測、finalizer invocation/error streamは、値を補完せず `not_independently_reconstructable_from_workspace` と記録する。unique outputsと内部exact-one countersからは `at_least_one_execution=true` とauthorization consumed/reuse forbiddenだけを確定する。
- current safety probeはread-onlyで、restore poststateと同じservice/worker/GPU/owner/hash/health、targeted process 0、新service epochを要求する。
- lock payloadの旧driver PIDをcurrent ownerとして扱わない。`/proc/locks` のkernel FLOCK holderがcurrent service main PIDだけで、生成時と同じdevice/inodeであることを要求する。cleanupはpassedへ書き換えず、substrate removalは未実施で、復旧serviceが同inodeを保有している間はunsafeと記録する。
- recovery mutationはruntime/capture/result/auditの封印に限定し、service、lock、GPU、processを操作しない。再呼出はpartial raw state preconditionで拒否する。
- targeted recovery testsは20件通過した。operator全体testはactual-v16 fresh rootsが既に存在するため旧pre-actual absence testで停止し、別担当の未commit maintenance source変更によりcurrent source authority testも現時点では通過不能だった。

## 次の行動

- source/tests/journal commitを独立QAへ渡し、実rootを変更せずpreconditions、source authority、live safetyを確認する。
- QAが通過した場合だけ `recover-failed-finalization` を一回実行する。`finalize-actual`、actual、maintenanceは再実行しない。
- 封印後は専用static validator、全root mode/SHA256SUMS/self-hash、Git blob/archive一致を独立監査する。
