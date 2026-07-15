# P3 resident downstream cascade v5

## 前回の要点

- READY candidate failure auditのrunner authorityは `076c3662`、capture authorityは `0e8bf9f4`、maintenanceのstrict consumer foundationは `87f3a5cb` で確定していた。
- binding-v5はroot mode契約漏れを修正したvalidator `a3f0527a` で再生成され、artifact commit `80d9b3a0`、root `0555`、members `0444` としてsealedになった。
- profile actual v3/v4の失敗証跡はhistorical immutable evidenceとして維持する必要があった。

## 今回の変更点

- launcherをsealed binding-v5へcascadeした。
  - launcher commit: `0ad6d0e63c877c30996c30f54b6c6b7eb84edf1f`
  - binding manifest SHA-256: `82a75a62a1b5ce254df5522f808ec1ffce00045265960342149eb9ddb29837d4`
  - runner: `076c3662` / `bb21d396...`
  - validator: `a3f0527a` / `8a151a4d...`
  - root mode `0555`とmanifestの`binding_root_contract`をfail-closedで検証する。
- execute-binding、profile run/evidence/capture、profile run-idをfresh v5 namespaceへ切り替えた。
- execute-binding-v5を生成した。artifact commitは `bd120fc5`。
- maintenanceをcurrent consumerへcascadeした。
  - maintenance commit: `bf2ef47a2a9f7d766a82b56b799b6646863d2c09`
  - launcher `0ad6d0e6`、runner `076c3662`、validator `a3f0527a`、binding `80d9b3a0`、capture `0e8bf9f4`をpinした。
  - base/profile ready、dry-run、maintenance、captureの出力をv5へ切り替えた。
- v3/v4 failureとready-v1のtestsをcurrent constantsから分離し、historical pathを明示した。test commitは `db116bd6`。
- QAを12 test filesで再計測し、533 collected / 533 passed / 0 failed / 0 deselectedへ更新した。
  - resident trust chain: 378
  - resident driver: 22
  - ROCTX ranges: 5
  - diagnostic capture: 49
  - selection raw producer: 26
  - profile family exclusion: 27
  - candidate selector: 26
- base/profile ready-v5と両dry-run-v5を生成した。artifact commitは `3a330dce`。
  - 全rootは `0555`、全memberは `0444` / nlink 1。
  - 両dry-runはpassedで、GPU command、model load、service touchはすべてfalse、全process countは0。

## 検証

- resident trust chain: `378 passed`
- cargo resident driver: `22 passed`
- diagnostic capture: `49 passed`
- remaining Python suites: `84 passed`
- QA aggregate: `533 passed`
- binding-v5、execute-binding-v5、base/profile ready-v5、両dry-run-v5の`SHA256SUMS`はすべてPASS。
- historical profile execute/capture v3とprofile evidence/maintenance/capture v4の`SHA256SUMS`はすべてPASS。
- profile actual/evidence/maintenance/capture/operator/result/audit v5はすべてfresh absenceを維持した。

## 次の行動

- actual profile diagnosticを行う場合は、fresh operator authorizationを別作業として作成し、明示承認後に最大1回だけ実行する。
- actual実行前にv5 output absence、quiet window、service/GPU ownership、ready/dry trust chainを再検証する。
