# P3 profile operator manifest v10

## 前回の要点

- profile-ready-v12はcommit `5456117e223653155897eaab9c176a2424198250`、tree `418af4a8f43ab4f58c306f66323e46d00cacc394`で確定した。ready binding SHA-256は`4c1fcee0c980e341e5346066a4a59bd7c8ace9eab562e18189b7050ceaf52890`、ready SUMS SHA-256は`c81139e9361b1a8ee740c3d0cb3202f333c5ccd88a4f766a9edd756a54fba575`。
- execute-binding-v9はcommit `dc9c12b6d9abb42edf52dccd1691b25cb83b0a47`。旧quiet-window-v14、operator-command-v9、operator-result-v9、actual-audit-v9は今回の作業前から封印済みだった。

## 今回の変更点

- operator-v10 source authorityをcommit `34de8502dc22dc02840129fb3fa0fc8d4d696308`、tree `65411e32294fcc908d85d640e4574c7fc257b646`、blob `da61fde1a80cf7b97f9cabfce8323fe6555b426c`、raw SHA-256 `3ca476bf9b1dfcf89e4f252ce0b6f2cab48c5f627cba66f51aa04ecb8bf4afb8`へ固定し、独立ready/operator GOを受領した。
- 初回read-only監査では、対象外の`/tmp/ullm-sq8-rust-v2-check/...`テストに属するPID `1328087`と`1334583`が外部AMD-SMI ownerとして検出されたため、quiet採取を開始しなかった。外部ownerが消失した後の直前監査ではproduction serviceがactive/running、main PID `1212941`、worker PID `1213021`、NRestarts 0であり、AMD-SMI/KFD ownerはworkerだけ、targeted process 0、fresh output 9/9 absentだった。
- fresh quiet-window-v15を1回採取した。27/27連続clean sample、span `317.155842855`秒、reset 0、最終confirmation passedでGOになり、全sampleのHEAD、tree、service epoch、owner、blocking identityが一定だった。
- quiet JSON SHA-256は`88c0bd727d2e18480e0faa716911b9c53ddea7365ba7d2e79648db9a99619c6a`、quiet SUMS SHA-256は`d8bd0d0885be2d6e6e6ca83d8a49589b9e15417ad6880e91b01be31137110f1e`。
- fresh operator-command-v10を1回生成した。ready-v12とquiet-v15をpinし、exact 10 argv、各authorization flag 1回、`shell=false`、maximum invocation 1、fresh outputs 9/9 absent、retry forbidden、outer-finally restore 120秒を固定した。
- operator command SHA-256は`2f7c11196a99ac277b973f7f29efdab691c0b28db19ca583cd35ee6b28221598`、manifest raw JSON SHA-256は`05f457d3cf17cc57db50add9456714407c2a442b94f9a3aa567e5d594cc64cff`、semantic manifest SHA-256は`fa56a843bf8cb58df637b932e384f454e2617e22b7bfe35d912fd8af27656c0d`、operator SUMS SHA-256は`7cd59f443e66667ba05fc7e1e2fb95326f8b60eda62ce2a3987d367bba8821c3`。
- 両rootは`0555`、全filesは`0444`かつnlink 1で、各`SHA256SUMS`とformal validatorがpassedした。対象テストは15/15 passed。旧quiet-window-v14、operator-command-v9、operator-result-v9、actual-audit-v9、ready-v12、execute-binding-v9は不変だった。
- quiet採取とoperator manifest生成はread-onlyで、GPU command、service操作、actualは実行していない。

## 次の行動

- quiet-window-v15とoperator-command-v10 artifact commitを独立QAへ渡し、archiveとGit blobの一致を再確認する。
- actualは別の明示的GOを受領し、直前のfresh 9/9 absenceとcurrent service identityを再確認するまで実行しない。
