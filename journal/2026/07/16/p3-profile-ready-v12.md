# P3 profile ready v12

## 前回の要点

- profile-ready-v11とdry-run-v11はcommit `abcf95ad3e56010b3e5f8b38c883c25bf5e2c780`でimmutable化した。旧v7〜v11とactual-v9は削除や上書きをしない。
- execute-binding-v9はcommit `dc9c12b6d9abb42edf52dccd1691b25cb83b0a47`で確定した。

## 今回の変更点

- maintenance-v12 final authorityをcommit `62e8fe91e073575c4776603786f9909f2b8001cd`、tree `22290d1aa488434ad29d2cf4ae7a17e679904fce`、harness blob `bffd039b86fdbb5d3cff7402e30f8b12f7ab2e1b`、raw SHA-256 `3295e56fba8b5139ffca55cc3d742d83a916aa8cb1cf53ded4f7f41fb268892d`へ固定した。
- final QA aggregateは639/639 passedで、manifest対象は12 test files。maintenance authority単体は156件passedだった。
- launcherはcommit `7f961f8de75ccbb1080fcd35a5b274584d4e00f3`、captureはcommit `1aed601a7e4102c99550b09384ef45fe57d43287`、familyはcommit `e4f8583a0fc710d2146f70d06b8b49eb42f04a16`、selection raw producerはcommit `c8becac66551f216de47d0cd935929afe60b3b96`へ固定した。
- fresh profile-ready-v12を1回生成し、formal loaderでstatus `ready_for_one_case`、execution mode `profile_diagnostic`、actual eligible true、authorization run ID `p2-r9700-resident-one-case-smoke-profile-diagnostic-v9`を確認した。
- fresh profile-ready-dry-run-v12を1回実行し、status `passed`、全process count 0、service操作、GPU command、model load、captureが未実行であることを確認した。
- ready binding JSON SHA-256は`4c1fcee0c980e341e5346066a4a59bd7c8ace9eab562e18189b7050ceaf52890`、ready SUMS SHA-256は`c81139e9361b1a8ee740c3d0cb3202f333c5ccd88a4f766a9edd756a54fba575`。
- dry-run evidence JSON SHA-256は`20834ed11e58b1d440c015d4bc38f4ab2fc6321dac6b2e86cdb44949d809a70e`、dry-run SUMS SHA-256は`87ebe1c6db8211bb3fa8818a5216d19e63237a27e42f1ee5513418febf75ebf3`。
- 両rootは`0555`、全filesは`0444`かつnlink 1で、各`SHA256SUMS`のreadbackがpassedした。旧ready/dry-run v7〜v11、actual/result-v9、execute-binding-v9は不変。
- v11 artifact testを固定pathのhistorical readbackへ変更し、v12 artifact testsと合わせて4/4 passedを確認した。
- GPU、service、actualは実行していない。

## 次の行動

- profile-ready-v12 artifact commitを独立QAへ渡し、archiveとGit blobの一致を再確認する。
- downstream authorityをprofile-ready-v12へrecascadeし、独立GOを得るまではGPU、service、actualを実行しない。
