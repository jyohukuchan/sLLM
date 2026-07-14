# P2 active fidelity gate attempt 3

## 前回の要点

attempt 2は、service停止後のRuntimeDirectory再作成権限不足でGPU開始前にexit 90となった。cleanupはserviceを復旧し、active/runningとidentity不変を確認した。

## 今回の変更点

sudo-nのRuntimeDirectory再作成を含むgate（SHA256 prefix `991c8b7f`）でattempt 3を開始した。service停止、RuntimeDirectory再作成、runtime lock取得、`RUN_STARTED` marker作成までは通過したが、homelab1 gateから `runuser -u homelab1` を呼び出したため `may not be used by non-root users` となり、exit 90で停止した。GPU binaryは起動していない。`RUN_STARTED`、service-stop marker、observer marker、`run.log`、`monitor.log` は保持し、run.logには失敗メッセージを記録した。

保存した証跡は `attempts/active-attempt3-20260715T083049Z/` に移し、次の5ファイルを `SHA256SUMS` で検証した。

- `monitor.log`: `23bca6d96ba5be75ea0342be8d4c1c92289dd37c98cf32382987c72b6115352d`
- `observer-sample.marker`: `088cc4e3c99b8ebde1244a0d10752035da66ef6cecd358aba24e81573237be35`
- `run-started.marker`: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- `run.log`: `c1cc14e19fa494034c45dd75bbd7bed66174aab7bddd04f4406a329f77c97662`
- `service-stopped.marker`: `2cacd09a93dff11c7747576a275fa752e3c1f84520391f9307e48e1a6eb009c7`

archive後のcleanupでBASE直下を空けた。post状態はservice active/running、MainPID `742209`、`NRestarts=0`で、runtime/lockおよびactive/package/worker identityは不変だった。

## 次の行動

binaryは現ユーザーのhomelab1として直接実行し、root専用操作は固定sudo配列、lockとbinary/output操作はhomelab1として維持する。修正中はGPU/serviceを再実行しない。次回実行はattempt 4として新しいarchiveを使う。
