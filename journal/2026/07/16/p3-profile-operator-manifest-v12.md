# P3 profile operator manifest v12

## 前回の要点

- profile-ready-v14/dry-run-v14はcommit `39af01d5dca7c76eb53fdbffc59dc976a2d24e6c`、tree `78a457d681ee23c43f87e0094ae60331635704c3`で確定した。ready binding/SUMS SHA-256は`6664abaafdf76adcc40565652dbbaa6ab0dbb1f131d1a4b011d66007fd059891` / `803046262d5b0d106ccecccb2979b3d8ff5d7d8bf4eece5b3a49f377f9c5b00d`。
- actual-v11 failure evidenceはcommit `854e5a348bd3c0f442f2371a0d3619308bce3b95`、tree `147bd97b595d8cea268c193e09e5c817ef6bdacc`で封印済みで、invocation 1/1、retry 0、`pre-stop-snapshot` failure、service/capture未到達だった。

## 今回の変更点

- operator-v12 source authorityをcommit `ad327b427a0cd4eed73078296316257f314b72c1`、tree `a580b3aa534ea74351fa4190a01899c723c8fd2a`、blob `b63a47c981be439bdd4f10535f26f9bec1f58dbd`、raw SHA-256 `3f6e72417ca1da67b154955e7bf52239dcaef97ceec89e952899daac9e58f210`へ固定し、独立ready/operator GOを受領した。operator testsは34/34、関連回帰は255 passed / 1 environment-dependent skipだった。
- 最初のquiet-v17 collectorは、採取中に対象外SQ8 test PID `1851777`がGPU ownerとして出現し、`restored worker does not uniquely own target GPU`で安全停止した。artifactは生成されず、quiet-v17/operator-v12 rootsはabsentのまま、GPU/service/actual操作もなかった。
- SQ8 parent shell/cargo/testと親tmux/codex配下をkill/signal/inputなしでread-only監視した。process群の自然終了後、3回の連続pollでAMD-SMI/KFD ownerがproduction worker PID `1213021`だけ、serviceがmain PID `1212941`、active/running、NRestarts 0、fresh roots absentで安定した。明示的な再採取GO後にだけ再開した。
- fresh quiet-window-v17を1回再採取した。27/27連続clean sample、span `319.078221845`秒、reset 0、最終confirmation passedでGOになり、全sampleのHEAD、tree、service epoch、owner、blocking identityが一定だった。
- quiet JSON/SUMS SHA-256は`25ad254c1beead814f390948293b4d8e651dd12b9db40ae3314d6978dfccfb5e` / `0958bbb6e2aa004b115da88b43b6e81501e3946295d010ab0167ea41d4a49086`。
- fresh operator-command-v12を1回生成した。ready-v14とquiet-v17をpinし、command-v11を`authorized_then_invoked_once_pre_stop_failed`、actual-v11を`pre_stop_failed_sealed`、historical actual-v9を`executed_sealed`として固定した。exact 10 argv、各authorization flag 1回、`shell=false`、maximum invocation 1、fresh outputs 9/9 absent、retry forbiddenを維持した。
- operator command SHA-256は`a2e208d06e9f872d9dddd95999189b4f11e08d95eb873620d85b482563f6e046`、manifest raw JSON SHA-256は`5712168a29d708d0ce7578d81f15089fb1dbed400dbba84e55887a4ee0348944`、semantic manifest SHA-256は`5f9b5a8758fe1dd22446f88c140a5bed5738de440f253eedba2cb5a0668f5b27`、operator SUMS SHA-256は`641d83c39957967fdcb39abedea901b11bd8eb214fe587f8f08cf9a0a858f396`。
- 両rootは`0555`、全filesは`0444`かつnlink 1で、formal validators、各`SHA256SUMS`、secret pattern scanがpassedした。ready-v14、operator-command-v11、actual-v11は不変で、fresh v12 actual rootsは9/9 absentだった。
- quiet採取とoperator manifest生成ではGPU、service、actualを操作していない。

## 次の行動

- quiet-window-v17/operator-command-v12 artifact commitを独立QAへ渡し、archiveとGit blobの一致、source/ready/previous actual authority、fresh absenceを再確認する。
- actualは別の明示的GOを受領し、直前のservice epoch、AMD-SMI/KFD owner、fresh 9/9 absenceを確認するまで実行しない。
