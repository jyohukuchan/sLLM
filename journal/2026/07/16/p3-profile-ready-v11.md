# P3 profile ready v11

## 前回の要点

- profile-ready-v10とdry-run-v10はcommit `19dce84189765fbca03ddd99da2920feab0cbf6e`でimmutable化した。旧v7〜v10とactual-v8は削除や上書きをしない。
- ready-v11はmaintenance-v11 final authority通知後にだけ生成し、execute-binding-v8、launcher、capture、selection raw producerの確定authorityを消費する条件だった。

## 今回の変更点

- maintenance-v11 final authorityをcommit `7e6486b4055e72584fcd2dfd9a6251048d683906`、tree `9b79b597e218a412484e875a9a6c2a0cdce34e0e`、harness blob `a1a33bbb6249e6605ae73f2b3626e29777476b2d`、raw SHA-256 `e81ec8f6f93a32881293403abef8e4ee2338d43862972d416efb432c3715e0ac`へ固定した。
- final QA aggregateは623/623 passedで、manifest対象は12 test files。maintenance authority単体は156件passedだった。
- execute-binding-v8はcommit `ee7333cdbc1da23f24295fe6d32462feebc6467f`、launcherはcommit `b81066dbf86857afbeb0dc7d41493fdef680266d`、captureはcommit `a098ca53c1c3e5c16ec02a08013c55b82f18301c`、selection raw producerはcommit `dac045244d7609c42c2db1ea0f91aa707ffb717b`へ固定した。
- fresh profile-ready-v11を1回生成し、formal loaderでstatus `ready_for_one_case`、execution mode `profile_diagnostic`、actual eligible true、authorization run ID `p2-r9700-resident-one-case-smoke-profile-diagnostic-v8`を確認した。
- fresh profile-ready-dry-run-v11を1回実行し、status `passed`、全process count 0、service操作、GPU command、model load、captureが未実行であることを確認した。
- ready binding JSON SHA-256は`ef23daf6b8166abc98fa0a72a0eeeae86ab24b5b1747ff0018c4240398ba0c18`、ready SUMS SHA-256は`7bb6a891969ef73a3024aec370c8e38a245bb95e21711e0f1b6068cdfabf9217`。
- dry-run evidence JSON SHA-256は`b5a863514207ed7055689a9b26e839254ec5805c67cfb626352904121a0dcd2a`、dry-run SUMS SHA-256は`5f09fe28a036e0fe476e3c9d2fd1003dd52f775bb77711347feb10647002841b`。
- 両rootは`0555`、全filesは`0444`かつnlink 1で、各`SHA256SUMS`のreadbackがpassedした。旧ready v7〜v10、dry-run v7〜v10、actual-v8、execute-binding-v8は不変。
- GPU、service、actualは実行していない。

## 次の行動

- profile-ready-v11 artifact commitを独立QAへ渡し、archiveとGit blobの一致を再確認する。
- downstream authorityをprofile-ready-v11へrecascadeし、独立GOを得るまではGPU、service、actualを実行しない。
