# P3 profile operator manifest v9

## 前回の要点

- profile-ready-v11はcommit `abcf95ad3e56010b3e5f8b38c883c25bf5e2c780`、tree `5f140564964883a67c2c2d8af066e8eecb935b37`で確定した。ready binding SHA-256は`ef23daf6b8166abc98fa0a72a0eeeae86ab24b5b1747ff0018c4240398ba0c18`、ready SUMS SHA-256は`7bb6a891969ef73a3024aec370c8e38a245bb95e21711e0f1b6068cdfabf9217`。
- execute-binding-v8はcommit `ee7333cdbc1da23f24295fe6d32462feebc6467f`。旧quiet-window-v13とoperator-command-v8はcommit `46219af1ce52c6af3bd29d6e84a0297ab6301823`で確定している。

## 今回の変更点

- operator-v9 source authorityをcommit `5c02a6e36f165b1a413488d59cc673054228ac31`、tree `3bc04e001157b8cc6337f7dcfbe5828f59dc4254`、blob `0b23c63b4b314c32a4f34957c3baae151d45bd3f`、raw SHA-256 `27effb270a11f5d9d6a8ed6897dee8d036cd44a4f66c8e5619e54a17f01d0f8d`へ固定し、独立ready/operator GOを受領した。
- 直前read-only監査でproduction serviceはactive/running、main PID `872658`、worker PID `873053`、NRestarts 0だった。AMD-SMI/KFD ownerはworkerだけで、targeted process 0、fresh output 9/9 absentを確認した。
- fresh quiet-window-v14を1回採取した。27/27連続clean sample、span `312.301672447`秒、reset 0、最終confirmation passedでGOになり、全sampleのHEAD、tree、service epoch、owner、blocking identityが一定だった。
- quiet JSON SHA-256は`87799427cde1c74a05a434d41f1b135686b9f23b4ea78a6e9d2c7906326ac366`、quiet SUMS SHA-256は`18bffb1b6a8428e19a8f499ae35ec759c5d8c23c8e7a9e5a5a78b91c48ee047a`。
- fresh operator-command-v9を1回生成した。ready-v11とquiet-v14をpinし、exact 10 argv、各authorization flag 1回、`shell=false`、maximum invocation 1、fresh outputs 9/9 absent、retry forbidden、outer-finally restore 120秒を固定した。
- operator command SHA-256は`127bc83be11deaed96b0fa2ee30144ff9ad620b820258afff309391607e8c4a5`、manifest raw JSON SHA-256は`02660cd846ae95726716abcd164db7eb492d04947bb7dbde4ef2532139193870`、semantic manifest SHA-256は`51e91b6495ad3da319a7e3f6d244efebc00a27eb0277d0f93b44748050fb7d19`、operator SUMS SHA-256は`cfa86555950e90f7c49a298812fde410fcfcab1f47aaf620ac37bdd8575ba9fd`。
- 両rootは`0555`、全filesは`0444`かつnlink 1で、各`SHA256SUMS`とformal validatorがpassedした。旧quiet-window-v13、operator-command-v8、operator-result-v8、actual-audit-v8は不変。
- quiet採取はread-onlyで、GPU command、service操作、actualは実行していない。

## 次の行動

- quiet-window-v14とoperator-command-v9 artifact commitを独立QAへ渡し、archiveとGit blobの一致を再確認する。
- actualは別の明示的GOを受領し、直前のfresh 9/9 absenceとcurrent service identityを再確認するまで実行しない。
