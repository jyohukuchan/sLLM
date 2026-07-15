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
- validator authorityはroot sealing追補source/tests commit `a3f0527a200bfe14f81301789089c61b59047116`へ更新し、tree `becc037dbb3a4e91c10499ea3bc2cf9b7c011748`、Git blob `bec05f97ad9156ae49aab27ccda5c73fab346d8e`、raw SHA-256 `8a151a4d3b44c266a667c1e902d284b019ed282639fa5f0f6e5339de0488e5bc`をarchiveと照合した。
- resident driverはbinary SHA-256 `458b8603d6823a1c20ea93e7c0d757c8910f3c36c9a2a34ab536853c0c9e7d34`、source/provenance/build pinsを変更していない。
- v5生成時のsubprocess countはrunner 1、trusted validator 1、fake driver 1であり、model load、GPU command、service操作はすべてfalseである。root modeを含む生成後inventory SHA-256は`3cf2fc65e8a67c1135983f67671a5fea6be28c74481d375363a16cabba650aac`である。
- 初回v5生成ではroot modeがumask依存の`0775`になり、validatorもroot modeを固定していない契約漏れをread-only follow-upで検出した。generator、validator、manifestへexact `0555`契約を追加し、writable-root負例を追加したうえでofficial fresh再生成した。最終v5はroot `0555`、全7 files `0444` / `nlink=1`である。既存binding-v4の歴史成果物は変更していない。
- prepared-v1は再生成せず、`SHA256SUMS`と正式validatorをreadbackした。modeを含むinventory SHA-256は生成前後とも`ba399b9e19e355cde0f16b1b37211515f83c5bb8e94c2a90b4221eb8828c7155`である。
- 既存binding-v4も変更せず、modeを含むinventory SHA-256は生成前後とも`ac7a5dc03486f7dafb010cd277e58fedc6cfd1b779176a21dcba24d41a7f02c4`である。
- v5の`SHA256SUMS`、formal `validate-binding`、runner/validator archive対Git object/raw照合はすべてPASSした。
- scoped testはgenerator `66 passed`、runner `53 passed`、READY candidate/capture `19 passed, 30 deselected`である。
- generator、runner、captureの統合pytestは`163 passed, 1 skipped, 3 failed`だった。三件は所有外launcherが旧validator SHA-256 `f11394f8...`を固定しているため、更新後source `c22e0586...`を意図どおりfail-closedで拒否したdownstream pin差分である。profile、ready、launcher、maintenanceは変更していない。

## 次の行動

- launcher/downstream laneはbinding-v5 manifestとvalidator authority `a3f0527a...` / `8a151a4d...`を明示的にpinし、該当するlauncher capture testsを再生成後に再実行する。
- actual、GPU、service実行は新しいsingle-use authorizationと全downstream pin更新が完了するまで行わない。
