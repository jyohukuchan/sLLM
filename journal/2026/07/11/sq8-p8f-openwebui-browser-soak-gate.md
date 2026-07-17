# SQ8 P8-F OpenWebUI 20-chat browser soak gate

日付: 2026-07-11

状態: 実装、独立review、R9700実機gateは完了

## 前回の要点

OpenWebUI Stop gateでは、実UI操作、Socket.IO相関、gateway lifecycle observer、systemd journal、service identity、秘密値を含まないatomic evidence bundleを固定した。P8-F release契約には、このStop caseとは別に、実OpenWebUIを通る20件の逐次成功chatが必要である。

## 今回の変更点

- `uLLM-project/deploy/openwebui/browser-soak.cjs`を追加した。Playwrightを1 process、1 contextだけ起動し、20件それぞれで新しい`temporary-chat=true`ページを1枚だけ作成して閉じる。
- 各caseは`navigate`、`select_model`、`submit_chat`、`wait_visible`、`wait_ready`の5 actionを順番に実行する。固有markerのUI表示、相関したSocket.IO content/done、入力欄の再有効化、page error/cancel/provider errorが0件であることを検証する。
- browser stdoutは20件のcase recordと1件のsummaryだけである。prompt、response、token、URL、chat ID、message IDは平文で出さず、byte数とSHA-256だけを保持する。`chat:active`と`chat:outlet`はcontent 0、done falseだけを許可する。
- `uLLM-project/tools/run-openwebui-soak-gate.py`を追加した。既存Stop gateの厳格JSON、`SO_PASSCRED` observer、journal cursor検証、atomic writerを変更せず再利用し、Soak専用state machineでexactly 20 trace、maximum active 1、前release後の次admission、全releaseの`outcome=stop`と`reset_complete=true`を検証する。
- gateの前後でgateway service unit、MainPID、UID/GID、boot ID、restart countを再確認する。observer payloadとjournal lifecycle MESSAGEはbyte単位で一致させ、journal cursorの重複を拒否する。
- Playwright imageは`sha256:<digest>`または`name@sha256:<digest>`だけを受理する。成果物にはimage reference自体を残さず、そのSHA-256とcontent digestだけを残す。
- 成功時だけ`observer.raw.jsonl`、`service-journal.raw.jsonl`、`browser/browser-stdout.jsonl`、browser summary、gate summaryを同一stageからatomic publishする。公開前に全regular fileと子directoryを`fsync`し、symlinkや想定外typeを拒否する。失敗時はcontainer、observer socket、runtime snapshot、stageを削除する。
- `uLLM-project/tests/test_openwebui_browser_soak.py`と`uLLM-project/tests/test_run_openwebui_soak_gate.py`に23 testsを追加した。改ざん、extra/missing/overlap、done後content、release算術、重複chat/message相関、state event、cancel/provider/page error、secret、image tag、raw payload/cursor、final journal seal、atomic abortとsymlinkを検証した。
- 新規23 tests、Ruff check/format、strict mypy、`py_compile`、Node構文検査、`git diff --check`は合格した。独立reviewで見つかったdone後本文、release durationの偽装、journal stop中の追加eventによる偽陽性を改変testで閉じた。

## R9700実機結果

- 実行版はcommit `2092eab`。bundleは`/home/homelab1/datapool/openwebui-soak-formal-20260711-193714`にatomic publishした。`summary.json` SHA-256は`8f027aacbfc70347f1a648f0bedeca1b7b40e0e9e1a6e223a2d3a6814ca4f3aa`である。
- 実browserの20 temporary chatは20/20成功した。browser stdoutは21 records、actions 100件、Socket.IO events 300件、page/cancel/provider errorは0件だった。
- gatewayは20 admission / 20 release、maximum active 1、全releaseは`outcome=stop`、`cancel_reason=null`、`reset_complete=true`、`completion_tokens=10`だった。先頭admissionから最終releaseまで38.099 s、requestのadmit-to-release medianは1.205 sだった。
- observerとjournalは100/100 lifecycle payloadでbyte一致し、cursorも100件全て一意だった。gateway PID `712139`、`NRestarts=0`、OpenWebUI health=healthyのままである。observer socketと一時containerも残っていない。

## 次の行動

20件のnested browser actionとgateway/journal証跡をP8-F release sessionの`browser_action`および`gateway_event` recordへ変換する。その前に、残り4段階direct cancel gateの独立reviewと実機実行を閉じる。
