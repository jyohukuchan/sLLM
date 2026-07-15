# AQ4 profile actual-v11 pre-stop finalizer recovery

## 前回の要点

- actual-v11 の exact-one invocation は return code 1 で終了し、operator raw streams と maintenance-evidence-v9 が生成された。
- maintenance は `pre-stop-snapshot` で外部 GPU owner を検出して失敗した。service stop、launcher、capture、rocprof は開始されず、restore は `attempted=false`、`passed=true`、`post_start=null` だった。
- 従来 finalizer は touched-service の new epoch restore だけを受理し、pre-stop no-op restore を封印できなかった。

## 今回の変更点

- pre-stop no-op restore を独立分岐にし、failure stage/reason、全 process counts、service safety、null launcher/capture/lock substrate/cleanup、package exact-one hash、downstream execute/runtime/capture absence が完全一致する場合だけ受理する。
- recovery snapshot を通常の `capture_snapshot` から分離した。actual output の存在を許可し、sealed operator-v11 manifest に埋め込まれた previous-v10 invocation 0 authority を使い、post-actual に previous-v10 fresh absence を再評価しない。
- no-op recovery は quiet confirmation と同一の service/worker/GPU/owner/lock/hash/formal-health epoch、active/running、NRestarts 0、worker-only owner、targeted process 0 を要求する。touched-service の attempted=true/new-epoch restore 契約は維持した。
- sealed maintenance failure の SHA-256、stage、reason を immutable failure snapshot として保存する。pre-stop probe が外部 owner PID/raw record を保存していない事実を `unavailable_not_recorded_by_pre_stop_probe` と明示し、post-hoc owner diagnostics を normative evidence にしない。
- 修正 source commit/blob/SHA-256 を execution authority ではなく、既存 raw streams と sealed maintenance を回収する finalizer authority として result/audit の双方に固定する。
- positive no-op recovery、attempted=true 回帰、restore pass false/unknown、後段stage、service/launcher/capture/rocprof開始、service touched、cleanup生成、partial runtime、service/owner/lock/hash/health差の負例を追加し、operator tests は33件全て passedした。
- 実 maintenance evidence SHA-256 `616a1a7bb9de0109093387856d81e41fa1944eedeaf83a15ad89a1714cd81b66` は新しい no-op validatorを通過し、execute/runtime/capture rootsの不在も確認した。編集対象外の evidence は変更していない。

## 次の行動

- source commit後、その commit/blob/SHA-256 を evidence-recovery-only finalizer authority として独立確認する。
- existing operator streams と sealed maintenanceだけを用い、actualを再実行せず finalizer を1回実行する。
- 封印後は result/audit、restore classification、failure snapshot、phase-aware recovery snapshot、全SUMSを独立監査する。
