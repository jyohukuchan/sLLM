# P2 resident launcher-v12 / execute-binding-v12

## 前回の要点

- launcher-v11はsource path-last-change commit `4cd57c1c0da182224df15c842e072dcc2c4a1de0`、execute-binding-v11 artifact commit `9111b2a6c9479ebccb61a55641b5be52f86d5dda`として確定済みである。
- actual-v15はcommit `99faf0066b93eb021fa83bea1b1a0193d9a79fd4`の6 output rootsを消費済みであり、再利用できない。

## 今回の変更点

- normal execute binding/run/evidenceをv12へ、profile runtime/execute-evidence/captureをv12へ、次のactual maintenance evidenceをv13へ進めた。新しい7 rootsはactual-v15の6 rootsと完全に非重複である。
- prepared-v2、binding-v7、runner、validator、resident source/binary、served manifestの各authorityは変更していない。
- v11専用の履歴loaderを追加した。current v12定数を参照せず、v11 rootの`0555`、3 membersの`0444`/single-link/SHA-256、artifact commit/tree/root tree、archived launcher sourceを検証するため、current final stateから独立している。
- current binding loaderはlauncher sourceのGit path-last-change commitが`launcher-trust.json`のcommitと一致することを必須にした。
- source/tests authority commitは`780a68007d424e1cf3f53d4e60728161ce6d13d4`、commit treeは`bee76471ea0faee2a5c95aea0fa405f0620fe515`、launcher Git blobは`1f55f4ec02e1f41c04e988f18d3c92f9c01689d5`、raw SHA-256は`55977291b6300b9365e685b4482a3c5ba3c21eb7e5ce7eb777aa7440791dda8a`である。
- fresh execute-binding-v12 artifact commitは`9fdab4c5aa2c60813fbe9c0527ac0bdffa725044`、commit treeは`df386bdae2db9f641e1657e2570a975827814f49`、artifact root treeは`5f0b2b39ec7d07b6ab068d08739c84c73c043c1e`である。
- artifactは3 filesで、SHA-256は次のとおりである。
  - `execute-binding.json`: `7e507c95b0f967fe4de25daf696e49271be3ccdc8a8ea978d7312ac1346714c1`
  - `launcher-trust.json`: `c7b71009eab9bc17888534fe4dbf75238405e2c33858b5332f0401d4b684845f`
  - `SHA256SUMS`: `5bcc26b36d9c93a748b567201fbfce9bc7f9987f5bff0701190f7cfd47a637af`
- direct testsはsource commit前にartifact検査1件を除いて`81 passed`、artifact生成・commit後に全件`82 passed`だった。すべて逐次実行した。
- GPU workload、model load、service操作、maintenance実行、profile capture、operator、actualは実行していない。bindingの状態は`blocked_pending_live_preflight_and_qa`、`actual_eligible=false`のままである。

## 次の行動

- maintenance側はlauncher authority commit `780a68007d424e1cf3f53d4e60728161ce6d13d4`とexecute-binding-v12 artifact commit `9fdab4c5aa2c60813fbe9c0527ac0bdffa725044`をexact authorityとして取り込む。
- maintenance-evidence-v13、profile runtime-v12、profile execute-evidence-v12、capture-v12だけを次のfresh actual namespaceとして扱い、independent QAとlive preflightが完了するまではblocked状態を維持する。
