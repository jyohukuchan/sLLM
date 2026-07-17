# SQ8 P8-F OpenWebUI post-header failure gate

日付: 2026-07-11

状態: 実装、独立review、R9700実機gateは完了

## 前回の要点

- OpenWebUIの派生imageでは、providerのpost-header SSE errorをUIへ表示し、normal doneを送らず、`chat:tasks:cancel`で入力欄を再び有効にするpatchを配備済みだった。
- Formal Stop、20-chat browser soak、4段階direct cancelは実機で合格していた。
- 残っていた製品動作は、生成途中のworker fatal、systemd自動再起動、同一temporary chatでの回復を一つのatomic evidenceとして閉じることだった。

## 今回の変更点

### Formal gate

- commit `35908f1`で`tools/run-openwebui-failure-gate.py`とformal browser clientを追加した。
- browserは部分本文を表示した後にhost controlを待つ。hostはworkerをPID/starttime/pidfdで固定して`SIGKILL`し、UI error、Socket.IO error/cancel、error後content/done 0件、失敗画面を検証する。
- systemdの新しいgateway/worker identityとrestart差分1、Docker network内`/readyz` 200を確認してから、同じchatの別messageで回復markerを送る。
- token、URL、prompt、chat/message/request IDはtext証跡に平文保存せず、browser出力とhost出力を分離して検証後に再snapshotした。control/runtimeは成功bundleへ残さない。
- 独立reviewでcontrol read-only化、browser出力競合、context close後event、empty content、journal priority、interim prefixなど6件を修正した。最終22 tests、Ruff、strict mypy、`py_compile`、Node構文、差分検査が合格した。

### 1回目の誤判定と修正

- 1回目はworker fatalとsystemd restart 0→1、UIのprovider errorまでは実際に成功したが、gateはatomic bundleを発行せず失敗した。
- 原因は`journalctl -u ullm-openai.service`が返すPID 1のsystemd管理行を、gateway process行と同じ`_SYSTEMD_UNIT=ullm-openai.service`として検証したことだった。実管理行は`_SYSTEMD_UNIT=init.scope`、`UNIT=ullm-openai.service`、`SYSLOG_IDENTIFIER=systemd`である。
- commit `5f4c359`で両者をtrusted journal metadataにより区別し、systemd行をlifecycleへ混ぜずに保存するよう修正した。当時の実journal 14行を再生し、gateway lifecycle 3行とsystemd管理6行を含めて合格した。
- commit `83f1236`で、固定内部文言だけの`FailureGateError`は失敗段階をstderrへ出し、秘密を含み得る任意例外は従来どおりgeneric表示だけにした。

### 正式実機結果

- atomic bundle: `/home/homelab1/datapool/openwebui-failure-formal-20260711-195916`
- `summary.json` SHA-256: `920486a8f209710ff2bfb10113cc09120ccbf0ee4e506f842d5755aef815bc35`
- screenshot SHA-256: `f5cfac6cd9b85bca472c09b088918b8a4aae525d57421724f789377a35d6fc0f`
- initial gateway/workerは`1421942/1422312`、recovered gateway/workerは`1452201/1452625`、restartは1→2で差分1だった。
- signal→`worker_fatal`は29.688 ms、fatal→browser errorは19.009 ms、error→Socket.IO cancelは8.217 msだった。
- error 1件、cancel 1件、error後done/content 0件、page error 0件だった。失敗画面には部分本文とtoast/inlineの`The generation failed.`が表示され、入力欄は有効だった。
- fault完了→recovered `/readyz` 200は53.753秒だった。readinessをhostではなく固定Docker imageからdeployment network内で確認した。
- 同じchatの新messageは`FAILURE_RECOVERY_OK`と一致し、done 1件、error/cancel 0件だった。gateway releaseは`outcome=stop`、5 completion tokens、`reset_complete=true`だった。
- browser action 9件、Socket.IO event 42件、gateway lifecycle 10件、journal 21件/cursor 21件である。stderrはbrowser/journalとも0 byteだった。
- 成功後はgateway PID `1452201`、worker PID `1452625`、`NRestarts=2`、OpenWebUI healthとgateway readinessはともに正常で、observer socketと一時containerは残っていない。

## 次の行動

1. standalone bundleはrunner回帰用pilotとして保持し、最終証跡へは結合しない。
2. 同一campaignの順序を合成証跡の独立validatorで先に固定し、検証済みrunnerを正しい順で再実行するorchestratorを作る。
3. batchなし・単一active requestのまま、通常100件、同じ唯一のfailure/restart、再起動後20件を一つのrunで収集する。
4. fixed fixtureのTTFT/decode matrixと最終publicationを閉じる。
