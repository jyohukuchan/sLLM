# AQ4 P3 profile operator v8

## Authority と version cascade

- profile-ready-v9: commit `ed10d543bcc0d196b157fad306a6995782692fe9`, tree `ef66fc9e696c821081542ef94e2cd68017ae272c`, ready blob `5c39f45b767dd62b7221482408cd38fde6050f46`
- ready JSON SHA-256: `fc05c02cb0d3eabc91ef08d31ea643d582cd615ecc4558031cca2b3af8fc5c5d`
- ready `SHA256SUMS` SHA-256: `c4ce3a86ad02c6252170b8f1b753d60a9dd011322141d92b20f2d7538c0c0570`
- quiet window は v13、operator command/result/actual audit は v8、profile runtime/execute evidence/maintenance evidence/capture は v7 を使用する。
- previous operator command v7 は immutable historical readback として参照する。ready-v8 は invalid-preoperator のため downstream authority に使わない。

## Finalizer

- exact-one、`shell=false`、retry forbidden、outer-finally restore の境界を維持した。
- return code 0 は complete diagnostic success、非0は failure evidence として分岐し、どちらも operator result、actual audit、runtime、capture を checksum 付き read-only tree に封印する。
- success capture の `measured-runs/` を含む再帰的な checksum coverage を追加した。

## Verification

- operator/finalizer tests: 10 passed
- print-actual: exact 10 argv、single `--confirm-one-case`、non-shell command を確認
- GPU、actual、service 変更は実行していない。
