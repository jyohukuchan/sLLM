# P2 active fidelity gate attempt 4 and served worker nlink audit

## 前回の要点

attempt 3は、非root gateからの `runuser` 呼び出しでGPU binary直前にexit 90となった。run開始markerと失敗ログをarchiveへ移し、serviceは復旧済みだった。

## 今回の変更点

attempt 4の保持証跡を `attempts/active-attempt4-20260715T083823Z/` へ移し、SHA256SUMSを検証した。run.logは `AQ4 fidelity capture failed: served worker must be a single-link regular file` であり、workerのnlink契約違反をbinary preflightが拒否した。captureのGPU処理へは進んでいない。BASE直下のrequire_absent対象9件はarchive後にすべて不在である。

served workerを読み取り監査した。

- path: `target/reasoning-v2/release/ullm-aq4-worker`
- owner/mode: `homelab1:homelab1` (uid/gid 1000), `0775`
- device/inode/nlink: `66306:10506405`, nlink `2`
- size: `3729520` bytes
- SHA256: `177f3106414efc7cc4b08fa2d87bed6e147d4188e0a290f43b7a1ac591fae48d`
- same-inode path: `target/reasoning-v2/release/deps/ullm_aq4_worker-03e49ec754c21dc7`

`/etc/ullm/served-models/active.json` は上記 reasoning-v2 worker pathと同じSHAを参照する。`deploy/served-models/qwen35-9b-aq4-reasoning.profile.json` も同じpathを参照する。installed `ullm-openai.service` のExecStartはgatewayであり、workerはactive manifestからgatewayが起動する。captureとworkerのsingle-link契約は、ハードリンク別名を残したmutable build成果物を実行しないためのidentity guardである。

production worker path、service、GPUは変更していない。安全なdetach手順は、同一ディレクトリでtemp copyを作成し、元のmode/ownerを適用して期待SHA・regular file・nlink1を検証した後、`mv -T`でworker pathだけをatomic renameする。実行中processは旧inode（deps alias）を保持し、次回service startから新inodeを読む。rollbackは保存した旧inode alias `target/reasoning-v2/release/deps/ullm_aq4_worker-03e49ec754c21dc7` から同じSHAをtempへ戻して検証後にatomic renameする。今回はこのdetach/rollback操作を実行していない。

## 次の行動

parentの明示判断と同一PTYでのsudo準備後だけ、worker detachを実施する。実施時は旧aliasのinode/SHA、temp copy、atomic rename前後のnlink、service復旧後のactive manifest/worker SHAを証跡化する。GPU captureはworker identityがnlink1であることをread-only preflightが確認してから検討する。
