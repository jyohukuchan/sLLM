# SQ8 calibration NO-GO 最終固定

- 前回の要点: immutable target 24行と独立BF16 sourceのcomparisonで、logits relative-L2が20/24行で固定上限1.0を超えた。固定policyはこの状態を集計前に病理的ドリフトとして拒否するため、calibrationはNO-GOである。
- 今回の変更点: 既存schemaを監査し、promotion failure receiptはactual前、holdout failure receiptはattempt消費後、freeze receiptはcalibration通過後専用であり、今回へ流用できる公式failure pathが無いことを確認した。metrics、freeze、holdout ledgerを作らず、専用の `ullm.qwen35_aq4_sq8_fidelity_calibration_rejection_evidence.v1` finalizer/validatorを追加した。
- evidence: `/tmp/ullm-p2-sq8-calibration-actual-17581136c57bb90e-v4/calibration-no-go-evidence-v1`。receipt SHA-256は `9e619f85683409a1addb04da71182b355638b53de91c1c9d71ca8db1ba789be6`、`SHA256SUMS` SHA-256は `ce93213db625b8666d41a19c3031754cdf2fe3f09c69c2ced065163d25ddad41`。directoryは0555、2ファイルは0444/nlink=1である。
- 固定内容: plan/actual/source/target/comparison、capture commit/tree、staged binary、gate script/logのcapture完了後rc=2、offline validator修正後の24-row valid、停止中stable2、service正常復旧、validator/tensor-authority/finalizer commit lineageを再検証する。観測はlogits relative-L2 20/24超過、min `0.9636891660654361`、mean `1.0767630755883828`、max `1.2498999406944615`、greedy mismatch 23/24、minimum top-k overlap 0、nonfinite 0である。
- 検証: NO-GO validatorと`sha256sum -c`はpass。focused finalizer testsは9 passed、calibration関連Python回帰はexit 0、Rust capture testsは11 passed。
- 次の行動: holdoutは `not_started / remaining 1` のまま実行禁止とする。P3は指定commit/worktreeをread-only監査し、このP2 NO-GOを上位selectorとproduction bindingの入力状態として扱う。
