# P2 resident launcher / execute-binding v11

## 前回の要点

- launcher-v10はsource authority `fc4559ee4fb8c7c1e62353fb3978a1a1e0a7d86d`、execute-binding-v10はartifact authority `2b477ed0dd1344d368e684e413cb756706af22f3`のhistorical artifactとして固定されている。
- actual-v14はprofile execute/capture/maintenance/operator/auditのv10/v11/v14 namespaceを占有済みで、archive commit `a2fe1ebac5d631919ca9082e17fda2126759a385`以後は再利用しない。
- runtime/prepared/binding authorities、resident binary `d7458f`、runner、validator、helper pinsは変更しない。

## 今回の変更点

- normal execute-binding、normal execute/result evidence、profile execute/result evidence、profile captureをfresh v11へ進めた。次maintenance lane用のmaintenance-evidence-v12を含む7つのfresh pathは、actual-v14が占有する6 pathsとdisjointである。
- launcher source/tests authorityはcommit `4cd57c1c0da182224df15c842e072dcc2c4a1de0`、tree `ba35d4e0642450a5c832f5f1d3fb526cc3911e27`、launcher blob `de145057e67b581963570b63adb12f167afb03fa`、raw SHA-256 `d0d7804d55b33754534501db4731581e742381f409b0ef290da4cc8db7949dcc`である。
- execute-binding生成時のlauncher authorityを共有HEADではなくlauncher sourceの最終変更commitへ固定した。loaderはcommit/tree/blobとcurrent source bytesをGitから再検証する。
- normal execute-binding-v11を1回生成した。statusは`blocked_pending_live_preflight_and_qa`、`actual_eligible=false`であり、run IDは`p2-r9700-resident-one-case-smoke-execute-v11`である。
- execute-binding JSON、launcher-trust JSON、SHA256SUMSのSHA-256は順に`ef8962ada001ef9017b76eb91fd9a89473b931aac857282296763750c5f9eb20`、`3c56816b7c07ae03c79f4670855137b7a9c37c9f637659695f47e5c581bc07c0`、`59146edcaa6b455d520783dc9e39dd096478f5414789c5056afc9d51506a68cf`である。
- artifact authorityはcommit `9111b2a6c9479ebccb61a55641b5be52f86d5dda`、repository tree `3102beeac6be9a7e03e871fb58a0476dd4115384`、artifact root tree `f76c878764aff5d4290bc48967928c0d1e1f6bac`である。rootは`0555`、3 filesは`0444`かつnlink 1である。
- direct testsはjobs 1で`81 passed`。profile direct testsは各テスト専用の一時lockを使い、`/run/ullm/r9700.lock`の終了状態に依存しない。
- historical execute-binding-v10は固定SHAとGit artifact bytesで不変を確認した。actual-v14の6 occupied pathsも`git diff --quiet a2fe1eba -- <paths>`で不変を確認した。
- GPU、service、model load、actualは実行していない。capture、maintenance、operatorのsource/artifactも編集していない。

## 次の行動

- execute-binding-v11 authorityをmaintenance/ready/offline laneへ渡す。
- live preflightと独立QAが揃うまではnormal execute-binding-v11をblockedのまま維持し、actualを実行しない。
