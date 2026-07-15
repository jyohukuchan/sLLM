# P3 profile operator manifest v11

## 前回の要点

- operator-command-v10はcommit `d278a2ba71a0f30c56c7af8927990eb4d6ac1e26`、tree `5a4d1b0a3a0e30c4befaef2f6e2cf355b3af3484`で確定した。manifest raw SHA-256は`05f457d3cf17cc57db50add9456714407c2a442b94f9a3aa567e5d594cc64cff`、semantic SHA-256は`fa56a843bf8cb58df637b932e384f454e2617e22b7bfe35d912fd8af27656c0d`、SUMS SHA-256は`7cd59f443e66667ba05fc7e1e2fb95326f8b60eda62ce2a3987d367bba8821c3`。
- profile-ready-v12はcommit `5456117e223653155897eaab9c176a2424198250`、tree `418af4a8f43ab4f58c306f66323e46d00cacc394`、execute-binding-v9はcommit `dc9c12b6d9abb42edf52dccd1691b25cb83b0a47`で確定している。
- actual-v10の直前監査ではproduction worker PID `1213021`に加えて、対象外テスト`/tmp/ullm-sq8-main-integration/...`のPID `1418316`がAMD-SMI ownerとして検出された。契約どおりsealed argvは実行せず、invocation 0、fresh 9/9 absent、result/audit absent、service操作なしで停止した。

## 今回の変更点

- 外部owner PID `1418316`の自然終了を確認した。kill、restart、GPU command、service操作は行っていない。終了後のread-only監査ではproduction serviceがactive/running、main PID `1212941`、worker PID `1213021`、NRestarts 0で、AMD-SMI/KFD ownerはworkerだけ、targeted process 0、fresh output 9/9 absentだった。
- operator-v11 source authorityをcommit `fbe46895015edce56377e5435cc2ef898f87b190`、tree `b4a0134c89b833cf2f42deeb0da2c5b5c0993f28`、blob `11354423f0042cefa2ee6f6a2641724cce2debf7`、raw SHA-256 `6e962546ea379146c4fa23f081436462ec3b1ccb8c8feec57490c598f4c2e67e`へ固定し、独立QA GOを受領した。対象テストは17/17 passed。
- fresh quiet-window-v16を1回採取した。27/27連続clean sample、span `317.689081904`秒、reset 0、最終confirmation passedでGOになり、全sampleのHEAD、tree、service epoch、owner、blocking identityが一定だった。
- quiet JSON SHA-256は`0439512b88211a1524110a8ac3724a4a8ba16bbb0b78370b2b64b4d650ab76e0`、quiet SUMS SHA-256は`038cd9fdb142c0770f70aed13c5fba3ad438a4129e4f98f8f4de4c3e15a8f4fc`。
- fresh operator-command-v11を1回生成した。operator-v10を`authorized_not_invoked_preflight_blocked`、invocation 0/1、fresh 9/9 absentとしてpinし、ready-v12、quiet-v16、historical actual-v9 `executed_sealed`を固定した。exact 10 argv、各authorization flag 1回、`shell=false`、maximum invocation 1、retry forbidden、outer-finally restore 120秒を維持した。
- operator command SHA-256は`2f7c11196a99ac277b973f7f29efdab691c0b28db19ca583cd35ee6b28221598`、manifest raw JSON SHA-256は`4597826e0c876e3b51c756f65c99c2bb43ee395504b7fe9767eb324db1706102`、semantic manifest SHA-256は`623730860c878b7652138bf54b8582677c48a346544244d0ee327b811d4b9387`、operator SUMS SHA-256は`a3fcc93e45071224e880449e48e5471134f9f82a1f0dd6c8e77446f4f24e11d6`。
- 両rootは`0555`、全filesは`0444`かつnlink 1で、各`SHA256SUMS`とformal validatorがpassedした。旧operator-command-v10とhistorical actual-v9 artifactは不変だった。
- quiet採取とoperator manifest生成ではactual、GPU command、service操作を実行していない。fresh v11 actual pathsは9/9 absentのまま。

## 次の行動

- quiet-window-v16とoperator-command-v11 artifact commitを独立QAへ渡し、archiveとGit blobの一致を再確認する。
- actualは別の明示的GOを受領し、直前のfresh 9/9 absence、current service epoch、AMD-SMI/KFD ownerを再確認するまで実行しない。
