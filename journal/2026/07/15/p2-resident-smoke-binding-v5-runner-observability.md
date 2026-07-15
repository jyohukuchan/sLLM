# P2 resident smoke binding v5 runner observability

## 前回の要点

- `resident-one-case-smoke-prepared-v1`は歴史的bootstrap runner `3dc4aa612b6cfd87675d0bd9fe506426f43e64f9`をimmutable control memberとして保持している。
- 既存`resident-one-case-smoke-binding-v4`はactual runner `81ceebb13518f590b5dbf439cd00b35e508c1c3f`を束縛しており、既存consumerが参照するため変更しない。

## 今回の変更点

- fresh `resident-one-case-smoke-binding-v5`を生成し、actual runnerをREADY failure observability authorityへ更新した。
  - commit: `076c3662aad6a3c8c74b3875882df4b41c026de7`
  - tree: `8e6d8a7470cbc6f6c943393fc06dee1edd1f8dca`
  - Git blob: `c6f5f30a0a3bc64dca01787648f19bb74edc95f5`
  - raw SHA-256: `bb21d396b045187cf1c10b3a240db8dd6a4cf769d657dfbfa377e676dbcf85fb`
- validator authorityはsource/tests commit `dd17e7326fbff50afb87dd2de8bb991f6b1bdb17`へ更新し、tree `6fa00dc36105ef2d7a68e48a2ea1237f391f9088`、Git blob `5ed525fb12c3aa8f9289195981a21c8a10c294bb`、raw SHA-256 `c22e05869146b8937384a72d410abbba7a8bf2368d79486048042a058c0d8f02`をarchiveと照合した。
- resident driverはbinary SHA-256 `458b8603d6823a1c20ea93e7c0d757c8910f3c36c9a2a34ab536853c0c9e7d34`、source/provenance/build pinsを変更していない。
- v5生成時のsubprocess countはrunner 1、trusted validator 1、fake driver 1であり、model load、GPU command、service操作はすべてfalseである。生成後のinventory SHA-256は`1e93f1dd82f053f81a830eaa02e941c674f68365d97b5d7b719d4e29db5c666e`である。
- prepared-v1は再生成せず、`SHA256SUMS`と正式validatorをreadbackした。modeを含むinventory SHA-256は生成前後とも`ba399b9e19e355cde0f16b1b37211515f83c5bb8e94c2a90b4221eb8828c7155`である。
- 既存binding-v4も変更せず、modeを含むinventory SHA-256は生成前後とも`ac7a5dc03486f7dafb010cd277e58fedc6cfd1b779176a21dcba24d41a7f02c4`である。
- v5の`SHA256SUMS`、formal `validate-binding`、runner/validator archive対Git object/raw照合はすべてPASSした。
- scoped testはgenerator `65 passed`、runner `53 passed`、READY candidate/capture `19 passed, 30 deselected`である。
- generator、runner、captureの統合pytestは`163 passed, 1 skipped, 3 failed`だった。三件は所有外launcherが旧validator SHA-256 `f11394f8...`を固定しているため、更新後source `c22e0586...`を意図どおりfail-closedで拒否したdownstream pin差分である。profile、ready、launcher、maintenanceは変更していない。

## 次の行動

- launcher/downstream laneはbinding-v5 manifestとvalidator authority `dd17e732...` / `c22e0586...`を明示的にpinし、該当するlauncher capture testsを再生成後に再実行する。
- actual、GPU、service実行は新しいsingle-use authorizationと全downstream pin更新が完了するまで行わない。
