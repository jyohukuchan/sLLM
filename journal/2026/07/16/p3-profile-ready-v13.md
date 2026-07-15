# P3 profile ready v13

## 前回の要点

- actual-v11 failure evidenceはcommit `854e5a348bd3c0f442f2371a0d3619308bce3b95`、tree `147bd97b595d8cea268c193e09e5c817ef6bdacc`で封印された。invocation 1/1、retry 0、returncode 1で、`pre-stop-snapshot`失敗のためlauncher、capture、rocprof、systemctl stop/startは0回だった。
- profile-ready-v12はcommit `5456117e223653155897eaab9c176a2424198250`、tree `418af4a8f43ab4f58c306f66323e46d00cacc394`で確定していた。ready JSON/SUMS SHA-256は`4c1fcee0c980e341e5346066a4a59bd7c8ace9eab562e18189b7050ceaf52890` / `c81139e9361b1a8ee740c3d0cb3202f333c5ccd88a4f766a9edd756a54fba575`。

## 今回の変更点

- actual-v11後に出現した対象外SQ8 cargo/test群はkill/restartせず自然終了を待った。複数回のread-only確認で親codex配下の関連cargo/testがなく、AMD-SMI/KFD ownerはproduction worker PID `1213021`だけ、serviceはmain PID `1212941`、active/running、NRestarts 0で安定した。
- minimal maintenance recascade authorityをcommit `576ab7d30f04742f4d48a200beb2e905b6ff83a9`、tree `f00a9380a901f63fde70fd6a647c334ba3250f1e`、blob `e177fc8e95a051c3d9370b7cec0729ab4c89dc2d`、raw SHA-256 `6c5a49e82ea4f00163bce9d7edbfaf511ed3a78e3bade98b194234ee9cbb8187`へ固定し、独立QA GOを受領した。pre-generation QAは639/639 passed。
- source namespaceはprofile maintenance evidence v10、profile-ready-v13、profile-ready-dry-run-v13へ更新し、execute/runtime/captureはv9を維持した。actual-v11の3 sealed rootsと3 downstream absent rootsをfail-closedでreadbackする。
- fresh profile-ready-v13を1回生成し、fresh dry-run-v13を1回実行した。readyは`ready_for_one_case`、`profile_diagnostic`、actual eligible、maximum invocation 1、output no reuseを維持した。dry-runはpassedで、全process count 0、service/GPU/model/capture操作なしだった。
- ready binding、harness trust、QA attestation、ready SUMS SHA-256は順に`d919d4addbda6338e7869ac185eeb47634e1da9d76793b5127357b638f31ec22`、`f69eaa84af3dc9a1ba7dc696999ccbe6b7c63486bb823dda8c92c4897cbaf59d`、`7190779299dce3666a7757efeff53afbaf13bc0ecdb57b0d88faf0de88af3006`、`2ad6093cae677b897a868918bfb68b98ae299016c150166b2c65ab15641a4f74`。
- dry-run JSON/SUMS SHA-256は`09012bb0a8e2c3f879718e560798fa5475473986729d205b07f9d1b29fc1cf92` / `44d6e4bd039b98c20915b29096888ea1e2e7c95356c23620a6ab55aa16c20de1`。
- 両rootは`0555`、全filesは`0444`かつnlink 1で、ready semantic readback、各`SHA256SUMS`、secret pattern scanがpassedした。旧ready/dry-v12、actual-v11 3 roots、execute-binding-v9は不変で、maintenance-v10とexecute/runtime/capture-v9は未生成のまま。
- pre-generation freshness guardを含むmaintenance suiteはartifact生成前に156/156 passedした。生成後は155/156で、唯一のfailureはready-v13 absentを要求するpre-generation専用guardであり、生成済みartifactの異常ではない。
- このpost-generation 1 failureにより、v13 rootsは`invalid-preoperator`と判定した。sealed bytesは上書き・削除せず失敗境界として保存し、operator、quiet、actualのauthorityには使用しない。

## 次の行動

- invalid-preoperator v13 artifact commitを独立QAへ渡し、archiveとGit blobの一致、source authority、process-count 0、旧artifact不変性を再確認する。
- fresh ready-v14 sourceと独立GOを待つ。v13はdownstreamへ使用せず、actual、quiet、operatorも実行しない。
