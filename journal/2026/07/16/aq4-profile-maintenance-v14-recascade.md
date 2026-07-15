# AQ4 profile maintenance v14 recascade

## 前回の要点

- profile-ready-v13 と profile-ready-dry-run-v13 は maintenance source commit `576ab7d30f04742f4d48a200beb2e905b6ff83a9` から生成された。
- launcher、execute-binding、execute-evidence、runtime、capture はv9、fresh maintenance evidenceはv10のままである。
- v13 artifacts生成後、current ready/dryのabsenceを要求するmaintenance testだけがfinal-stateで再現不能になった。

## 今回の変更点

- v13 ready/dryを commit `5f67d7edf9ea6285b6b5c01445b3dadbca65d562`、tree `6c01686cfa456ce17b34646627682b3afe8d59d1` のsealed invalid-preoperator readbackとして固定した。
- ready-v13 raw SHA-256 `d919d4addbda6338e7869ac185eeb47634e1da9d76793b5127357b638f31ec22`、SUMS SHA-256 `2ad6093cae677b897a868918bfb68b98ae299016c150166b2c65ab15641a4f74`、dry-v13 raw SHA-256 `09012bb0a8e2c3f879718e560798fa5475473986729d205b07f9d1b29fc1cf92`、SUMS SHA-256 `44d6e4bd039b98c20915b29096888ea1e2e7c95356c23620a6ab55aa16c20de1` と全Git blobを検証する。
- dry-v13 process count 0、service/GPU/model-load/capture未実行、ready-v13のlauncher-owned v9 output bindings、harness/QA authorityをreadbackする。
- current ready/dryのabsence assert 2件は完全削除した。artifact生成前後で共通のfresh safetyとしてmaintenance-v10、execute-v9、execute-evidence-v9、capture-v9のabsence 4件だけを維持した。
- current ready/dry namespaceをfresh v14へ更新した。launcher、execute-binding、runtime、execute-evidence、capture v9とmaintenance v10は不変である。
- source/test namespace変更は commit `59b78b4af7bb574a456c238f5f8ecf790df9ea0c` に保存し、maintenance test blob `00c1dfe57c869d516e932e6c6b3ac243ce3fbaec`、count 156をQA exact test manifestへ反映した。
- maintenance testsを含むresident trust chain 382件、historical artifact tests 2件、resident driver 22件、残りのQA Python suites 235件が全てpassedした。QA aggregateは639件、failed 0である。GPU command、service操作、actual実行、v14 artifact生成は行っていない。

## 次の行動

- final maintenance source authorityをprofile-ready-v14生成前に独立確認する。
- fresh profile-ready-v14とdry-run-v14を生成・封印した後も同じmaintenance testsを再実行する。
- operator recascadeではmaintenance-v10とready-v14だけを更新し、launcher-owned v9 output namespaceを維持する。
