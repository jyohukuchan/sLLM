# P3 profile operator manifest v8

## 前回の要点

- profile-ready-v10はcommit `19dce84189765fbca03ddd99da2920feab0cbf6e`で確定した。ready binding SHA-256は`cc4c9f76c7438c7e25a33db4bfa9c4b1de34ca2273f2b522de1dce52d3a65a61`、ready SUMS SHA-256は`59cc1c52d864040ba722ceb7a88bd4c0cf961b1d311912be79bffa55cccb4690`。
- operator source authorityはcommit `fdc276b9b520755d9005e96921611dc0a12ed6f1`、tree `4c2cab250962594875cb19068dd8ea5fb6f513ad`、blob `beed3e1c86e59b770ba7557aabfa40cecae42ba4`、raw SHA-256 `ef5542d1d95a41edf64ddd15dcd20e5c49854b09ac8864e728cac57fe4d7ecb6`で確定した。

## 今回の変更点

- 最初のquiet採取はservice epochがmain/worker `466848/467004`から`790940/791055`へ変化したため、約96秒で`running worker is not unique`としてfail-closedした。成果物は生成されなかった。
- 新epochに対する独立read-only監査2回と親GOの後、fresh quiet-window-v13を最初から再採取した。
- quiet-window-v13は27/27連続clean sample、span `311.965172889`秒、reset 0、最終confirmation passedでGOになった。serviceはmain `790940`、worker `791055`、active/running、NRestarts 0、AMD/KFD ownerはworkerのみで固定された。
- quiet JSON SHA-256は`11710608ee25f03b61d2df56eb71dcdc47b8e341ef1e186b860dd27cb9d89e67`、quiet SUMS SHA-256は`2d85c42f7913070bd650d6dc9c915e1f075375c2944b18aaa7bd10555d131b5a`。
- operator-command-v8はready-v10とquiet-v13をpinし、exact 10 argv、`shell=false`、maximum invocation 1、fresh outputs 9/9 absent、retry forbidden、outer-finally restore 120秒を固定した。
- operator manifest raw JSON SHA-256は`8eb7a31142a74efffe1a06bfd65e1d86f0d4343e196f0af8849a3d0421c947d1`、semantic manifest SHA-256は`80bbcf957f43f3c9d8cf8e3177a91b86df11eded66eb3174a9d87c9ef3d92c7c`、operator SUMS SHA-256は`5e5e03f653a5e1a9c6a1a5db40538485635bc60c32b5952cd81e9f424a943bbb`。
- 両rootは`0555`、全filesは`0444`かつnlink 1で、各`SHA256SUMS`とformal validatorがpassedした。
- 旧quiet-window-v12、operator-command-v7、operator-result-v7、actual-audit-v7は不変。actual、GPU command、service操作は実行していない。

## 次の行動

- quiet-window-v13とoperator-command-v8 artifact commitを独立QAへ渡し、archiveとGit blobの一致を再確認する。
- actualは別の明示的GOを受領し、直前のfresh 9/9 absenceとcurrent service identityを再確認するまで実行しない。
