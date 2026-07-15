# P3 profile operator sealed mode contract

## 前回の要点

- quiet-v18の収集前監査で、prepared-v2の`resident-driver`が`0555`である一方、operator verifierが全封印ファイルを`0444`と仮定していたため、実artifactを拒否した。
- `resident-driver`の内容SHA-256は`SHA256SUMS`と一致し、Gitのmodeも実行可能ファイルとして記録されていた。
- quiet-v18、command-v13、result-v13、actual-audit-v13は未生成のまま維持した。

## 今回の変更点

- prepared-v2とbinding-v7のartifact rootを明示し、役割・パス別のmode manifestを追加した。
- prepared-v2はroot `0555`、`resident-driver`だけを`0555`、その他memberを`0444`、各memberのnlinkを1として検証する。
- binding-v7はroot `0555`、全memberを`0444`、各memberのnlinkを1として検証する。
- 一律mode仮定を廃止し、inventoryにも実際に要求したmodeを記録する。
- 実prepared-v2/binding-v7を通る`audit-current`統合テストを追加した。
- `resident-driver`の`0444`/`0644`とJSONの`0555`を拒否するテストを追加した。

## 検証

- source/tests commit: `dd725b6db8b9c34a77995ea710a19ff5aad8b724`
- targeted: `4 passed, 40 deselected`
- operator full: `44 passed in 4.66s`
- `py_compile`: 成功
- 実`audit-current`: `status=clean`
- fresh outputs: `9/9 absent`
- service: `ullm-openai.service`, `active/running`, `NRestarts=0`, MainPID `2356631`
- production worker: PID `2357251`
- AMD-SMI/KFD owners: どちらも`[2357251]`
- actual execution、GPU workload、service stop/start: 0回

## 次の行動

- 外部SQ8 familyが不在で、production workerだけがGPUを所有する安定状態を維持していることを再確認する。
- quiet-v18は既定の完全窓で収集し、GO封印後にcommand-v13をexact-one pendingとして生成する。
