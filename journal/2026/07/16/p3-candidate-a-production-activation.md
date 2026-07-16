# P3 Candidate A production activation

## 前回の要点

- P3 rawはP2 qualificationに結合され、capability contractはexact fieldsになった。

## 今回の変更点

- selectionをrawから独立再計算し、Candidate A selected/eligible、build identity、profile raw refs、P2 `qualified_go`を結合するproduction activation toolを追加した。
- runtimeのdirect routeは環境変数だけでは有効にならず、diagnostic gateまたはworkerが検証したproduction activationを追加で要求する。
- workerにactivation CLIを追加し、activation/selection/qualification/rawのsingle-link identity、SHA-256、self/semantic hashとstatusを再検証するようにした。
- diagnosticとproductionの併用、direct requestだけの起動、activationなしproductionをfail-closedにした。
- selector publishをhard-link create-newにし、file/symlink/hardlink/8並列競合で上書きしないことを固定した。

検証結果: Python activation 8 tests、P3 Python combined 157 tests、selector 75 tests、worker P3 9 tests、runtime default-off testが通過。GPUとholdoutは実行していない。

## 次の行動

- P3 productionは実P2がNO-GOのためdefault OFFを維持する。
- 実際のqualified holdout artifactとP3 profile/selectionが将来揃った場合だけactivationを発行する。
