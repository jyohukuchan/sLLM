# P2 resident smoke launcher v10

## 前回の要点

- launcher-v9 sourceはcommit `7f961f8de75ccbb1080fcd35a5b274584d4e00f3`、execute-binding-v9はcommit `dc9c12b6d9abb42edf52dccd1691b25cb83b0a47`で確定している。
- profile actual-v12はcommit `44617f7fd46c39f71f04502b248739cc116fe095`へfailure evidenceとして封印され、v9以下のartifactとともに不変である。
- directional HIP copy runtimeを含むprepared-v2/binding-v7はartifact commit `eb96aa10c1b90bcbb3e457c76d7296ea3caaed44`、tree `bc5c2c07a8f74262bff7aa30676f8dbb2bcd2fed`で確定した。

## 今回の変更点

- launcher inputをprepared-v2/binding-v7へ更新し、input fingerprint `584ab574b6283b21f13216c265cd16c88abb836b1c94b2630127247215f633d1`、binding manifest SHA-256 `3b99dcfd11f9c4726a8531f9f828ec62dd84fabe577b6b529636ee0b66918579`を固定した。
- actual runner `d367b6da07393f55c720ded7250bda8cdc402a79`、prepared bootstrap `410d6fa1876a6772215604ba765ae1d6a91d67b9`、validator `e36a03ad423a0bb45cc1e4de67d3ca4fddfacdbc`の分離authorityを固定した。
- resident sourceをruntime commit `43ba16f2347a45caba8a60cac2189714118db280`へ更新し、binary SHA-256 `d7458fcdf8553871cac00123413676625c61eff2fdee3be9a440e656f05bcc1e`、3,505,000 bytes、Build ID `033ce9b214e2149861a8fcf0381c27bbac5bf1d1`、jobs=1のA/B再現buildを固定した。
- execute-binding、通常execute run/evidence、profile execute run/evidence/captureをv10 namespaceへ更新した。名前空間テストは将来のreadyやresultの存在を禁止せず、相互非衝突だけを検査する。
- launcher source authorityはcommit `fc4559ee4fb8c7c1e62353fb3978a1a1e0a7d86d`、tree `a5f938243463e36e401787aa62dfa6a5ef46e125`、blob `debace42c2063c476a9db3dcfe7fdf480bdf5088`、raw SHA-256 `5197efa84ec98343dda9438e4c0bc31e144765ce686a4b41199f1ae0315de8a6`である。
- fresh execute-binding-v10を1回生成した。execute-binding JSON SHA-256は`6fb8e61d4460ab89fdd643e917c7c20d1ddd9a68b1292703f0a2bd4d86ecef06`、launcher-trust JSON SHA-256は`33182ae19350cc7ed0a8fe3b439746a81996dc70a5d6d355fb0aac323e75dd6c`、SUMS SHA-256は`059ab6bab846f94b511a3d602a8cca350a328cf11b7dcf0f50a5ae8407b698de`である。
- artifactはstatus `blocked_pending_live_preflight_and_qa`、actual eligible false、launcher trust status `qa_pending`である。rootは`0555`、3 filesは`0444`かつnlink 1、formal loaderと`SHA256SUMS`がpassedした。
- launcher direct testsはartifact生成後に80/80 passedした。GPU command、service操作、actual、profile operatorは実行していない。

## 次の行動

- launcher-v10とexecute-binding-v10 authorityをmaintenance-v11／profile-ready-v15側へ渡す。
- 独立QA、ready、operatorの全authorityが確定するまではactualを実行しない。
