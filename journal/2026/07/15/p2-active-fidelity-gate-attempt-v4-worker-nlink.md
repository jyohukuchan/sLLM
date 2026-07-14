# P2 active fidelity gate attempt 4 and served worker nlink audit

## 前回の要点

attempt 3は、非root gateからの `runuser` 呼び出しでGPU binary直前にexit 90となった。run開始markerと失敗ログをarchiveへ移し、serviceは復旧済みだった。

## 今回の変更点

attempt 4の保持証跡を `attempts/active-attempt4-20260715T083823Z/` へ移し、SHA256SUMSを検証した。run.logは `AQ4 fidelity capture failed: served worker must be a single-link regular file` であり、workerのnlink契約違反をbinary preflightが拒否した。captureのGPU処理へは進んでいない。BASE直下のrequire_absent対象9件はarchive後にすべて不在である。

served workerを読み取り監査し、parentの明示判断に従ってsingle-link detachだけを実施した。

- path: `target/reasoning-v2/release/ullm-aq4-worker`
- owner/mode: `homelab1:homelab1` (uid/gid 1000), `0775`
- detach前のdevice/inode/nlink: `66306:10506405`, nlink `2`
- size: `3729520` bytes
- SHA256: `177f3106414efc7cc4b08fa2d87bed6e147d4188e0a290f43b7a1ac591fae48d`
- same-inode path: `target/reasoning-v2/release/deps/ullm_aq4_worker-03e49ec754c21dc7`

`/etc/ullm/served-models/active.json` は上記 reasoning-v2 worker pathと同じSHAを参照する。`deploy/served-models/qwen35-9b-aq4-reasoning.profile.json` も同じpathを参照する。installed `ullm-openai.service` のExecStartはgatewayであり、workerはactive manifestからgatewayが起動する。captureとworkerのsingle-link契約は、ハードリンク別名を残したmutable build成果物を実行しないためのidentity guardである。

同一ディレクトリでtemp copyを作成し、元のmode/ownerを適用して期待SHA・regular file・nlink1を検証した後、`mv -T`でworker pathだけをatomic renameした。detach後はworkerがdevice/inode `66306:10490132`、nlink `1`、aliasが旧inode `66306:10506405`、nlink `1`となり、両者のSHAは同一である。alias pathは現在も通常ファイルとして存在し、rollback copyは作成していない。実行中processは旧inode（deps alias）を保持し、次回service startからworker pathの新inodeを読む。serviceは停止・再起動しておらず、postcheckは`active/running`、MainPID `782385`、NRestarts `0`、`/run/ullm`とlockのidentityも維持している。active manifest、package manifestのSHAも維持している。rollback操作とGPU captureは実行していない。

## 次の行動

worker identityがnlink1となったため、Gateのread-only preflightを改めて確認できる状態になった。ただし、parentの追加判断があるまでGPU capture/Gateは実行しない。今回のdetach結果は`attempts/worker-detach-audit-v1/detach-result.json`と`rollback-binding.json`に固定した。
