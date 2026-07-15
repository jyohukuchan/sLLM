# P3 A4 canonical ready regeneration

## Scope

- A4のsealed FD map境界を、immutable launcherとmaintenance harnessの固定値へ反映した。
- `execute-binding-v1`、base `ready-v1`、profile `ready-v1`を、各出力が存在しないことを確認してから公式CLIで生成した。
- actual実行、GPU実行、サービス操作は行っていない。P2 attempt4の診断証跡は参照していない。

## Trust pins

- launcher: commit `868b9f72739bfb7aabc492ba5b4880b7f44e8b3d`, blob `e9f6a11131faeb7d766b85e5ef512839f4071be9`, SHA-256 `a62c054d7a0712a7aa1aedda71b38fae5378948dd6c9ad39ded70720e1ecea01`
- runner/capture: commit `ede2b872ab0de5550adbcb1b1dca8b4bbd789efd`
- validator: commit `ec25754440e4655750e3bc4ef11c0f1580dbf2f9`
- binding: commit `71fc08aadc6bc1a0a3aed85be3502c7362ad8e55`
- maintenance: commit `f71fa84f79eb6b863d3c1779303a993cfa73f0f5`, blob `32725371678755ba09e0e5686fbed8a371269034`, SHA-256 `d14b25d553ddfe8f40d794510a873ea134beff4a1a7ba8cd35ec6da0e2214741`
- B4 maintenance fix `407123e02593fd91c2c12672f8e6b811255789f2`はmaintenance commitの祖先である。

## Generated artifacts

- execute binding `SHA256SUMS`: `db9aedc0c5d759d5c3bd1f33235de463dd78655f058b7267cc382f8c59cdc195`
- base ready `SHA256SUMS`: `ea2de1a010e0ef25c4c0e6a8ac9f07ae6031d908b46845077c733b7d41fc332b`
- profile ready `SHA256SUMS`: `11128be05a14a1884fa5b73e86af50cce9e481c13efce42588d15ba926706fc8`
- 全ディレクトリはmode `0555`、全メンバーはmode `0444`である。
- profile readyには静的な`target-command-manifest.json`を含めない。対象manifestはlive preflight後に実行ごとに生成する。

## Verification

- launcher、execute、capture: `103 passed`
- maintenance（canonical base/profile dry-run、fake gateを含む）: `135 passed`
- prepare bundle、resident batch: `102 passed`
- 3つの`SHA256SUMS`は全メンバーで検証成功した。
- 独立read-only監査でも、3組のSUMS、launcher/maintenance readback、canonical testsが成功した。

## Boundary

- execute bindingは`blocked_pending_live_preflight_and_qa`、`actual_eligible=false`のままである。
- base/profile readyは明示的な1ケース実行を許可する準備証跡だが、この作業では実行していない。
- profileは診断専用で、`measurement_eligible=false`、`promotion_eligible=false`を維持する。
