# AQ4 P3 profile operator v8

## Authority と version cascade

- profile-ready-v10: commit `19dce84189765fbca03ddd99da2920feab0cbf6e`, tree `99b6ff5f4cc44809d4ba0b363ed395e5b8a5135b`
- ready JSON SHA-256: `cc4c9f76c7438c7e25a33db4bfa9c4b1de34ca2273f2b522de1dce52d3a65a61`
- ready `SHA256SUMS` SHA-256: `59cc1c52d864040ba722ceb7a88bd4c0cf961b1d311912be79bffa55cccb4690`
- quiet window は v13、operator command/result/actual audit は v8、profile runtime/execute evidence/maintenance evidence/capture は v7 を使用する。
- previous operator command v7 は immutable historical readback として参照する。ready-v8、ready-v9、operator source commit `9f3680ff` は invalid-preoperator のため downstream authority に使わない。

## Finalizer

- exact-one、`shell=false`、retry forbidden、outer-finally restore の境界を維持した。
- return code 0 は complete diagnostic success、非0は failure evidence として分岐し、どちらも operator result、actual audit、runtime、capture を checksum 付き read-only tree に封印する。
- success capture の `measured-runs/` を含む再帰的な checksum coverage を追加した。

## Verification

- operator/finalizer tests: 10 passed
- print-actual: exact 10 argv、single `--confirm-one-case`、non-shell command を確認
- GPU、actual、service 変更は実行していない。
