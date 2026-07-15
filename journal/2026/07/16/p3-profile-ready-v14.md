# P3 profile ready v14

## 前回の要点

- profile-ready-v13/dry-run-v13はpost-generation freshness guardが1件失敗したため、commit `5f67d7edf9ea6285b6b5c01445b3dadbca65d562`、tree `6c01686cfa456ce17b34646627682b3afe8d59d1`で`invalid-preoperator` evidenceとして封印した。v13はoperator、quiet、actualのauthorityには使用しない。
- actual-v11 failure evidenceはcommit `854e5a348bd3c0f442f2371a0d3619308bce3b95`、tree `147bd97b595d8cea268c193e09e5c817ef6bdacc`で封印済みで、invocation 1/1、retry 0、service/capture未到達だった。

## 今回の変更点

- maintenance-v14 authorityをcommit `3c5e6944a737888a7207d980b6a909f142f504f1`、tree `3cca7cf3f08d7857d6a140adb42bc10ec5044d4a`、blob `81af148923ebafa2b52a8f397ad71215bd88ce89`、raw SHA-256 `7aa6fc9ff72495d92dac4de10613badc0661c5f7508494e8346b4dbc5c2d7244`へ固定し、独立QA GOを受領した。QA aggregateは639/639、maintenanceとhistorical artifact testsは158/158 passed。
- v13 artifactは固定pathのhistorical readbackへ分離し、current ready/dry absence assertを削除した。runtime-v9、execute-evidence-v9、maintenance-v10、capture-v9のfresh absence guardは維持した。
- 直前read-only監査ではproduction serviceがmain PID `1212941`、worker PID `1213021`、active/running、NRestarts 0で、AMD-SMI/KFD ownerはworkerだけだった。fresh ready-v14/dry-run-v14とdownstream 4 rootsはabsentだった。
- fresh profile-ready-v14を1回生成し、fresh dry-run-v14を1回実行した。readyは`ready_for_one_case`、`profile_diagnostic`、actual eligible、maximum invocation 1、output no reuseを維持した。dry-runはpassedで、全process count 0、service/GPU/model/capture操作なしだった。
- ready binding、harness trust、QA attestation、ready SUMS SHA-256は順に`6664abaafdf76adcc40565652dbbaa6ab0dbb1f131d1a4b011d66007fd059891`、`c4081c1bce60b49323a8bda1e49ace2be98f9906de5bf23b09006c500f6c43ab`、`0e94d135475e618408c64b87403f95a1390a53989376fe55440500d249d2873d`、`803046262d5b0d106ccecccb2979b3d8ff5d7d8bf4eece5b3a49f377f9c5b00d`。
- dry-run JSON/SUMS SHA-256は`2b928f64012673a1e185767dbc32e296bd810e0c4f34242eb44d45f5bfbb3564` / `64567497603e31b764891c8ea91adc8457f9b1894dd4457bc7f77b8bb7699bb4`。
- 両rootは`0555`、全filesは`0444`かつnlink 1で、formal loader、各`SHA256SUMS`、secret pattern scanがpassedした。生成後testsは158/158 passed。旧v13とactual-v11は不変で、maintenance-v10、execute/runtime/capture-v9は未生成のまま。
- ready/dry artifact生成ではGPU、service、actualを操作していない。

## 次の行動

- profile-ready-v14/dry-run-v14 artifact commitを独立QAへ渡し、archiveとGit blobの一致、source authority、process-count 0、旧artifact不変性を再確認する。
- operator/quiet/actualは別の明示的GOと対応するrecascadeを受領するまで実行しない。
