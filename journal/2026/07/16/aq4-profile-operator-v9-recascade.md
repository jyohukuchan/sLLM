# AQ4 profile operator v9 recascade

## 前回の要点

- profile actual-v8 は rc 1、invocation 1/1、retry 0 の immutable failure evidence として commit `4b651cd5c46212349b5a598b344da6ea11993d30` に封印されている。
- maintenance、capture、launcher、producerの契約修正を反映した profile-ready-v11 は commit `abcf95ad3e56010b3e5f8b38c883c25bf5e2c780` に封印された。

## 今回の変更点

- ready authority を commit `abcf95ad3e56010b3e5f8b38c883c25bf5e2c780`、tree `5f140564964883a67c2c2d8af066e8eecb935b37`、ready SHA-256 `ef23daf6b8166abc98fa0a72a0eeeae86ab24b5b1747ff0018c4240398ba0c18`、SHA256SUMS SHA-256 `7bb6a891969ef73a3024aec370c8e38a245bb95e21711e0f1b6068cdfabf9217` に更新した。
- active execute bindingをv8、profile ready dry-runをv11へ切り替え、execute-binding-v8 artifact commit `ee7333cdbc1da23f24295fe6d32462feebc6467f` の全blobを検証するようにした。
- fresh namespaceをquiet-window-v14、operator-command-v9、maintenance/runtime/execute-evidence/capture v8、operator-result-v9、actual-audit-v9へ更新した。
- previous operator authorityをoperator-command-v8へ進め、v8をimmutable historical readbackとしてmanifestへ記録する。
- ready root全体、execute-binding root全体、trusted sources、fresh output 9 pathsをそれぞれGit blobとsealed checksumに結び付けた。
- finalizerはrc 0と非0の双方で、maximum/invocation 1、shell false、retry false、restore pass、retry-forbidden cleanup、success/failure evidenceのimmutable sealを検証する。
- operator/finalizer testsは11件すべてpassedした。GPU、service、actualは実行していない。

## 次の行動

- operator source commit後にtrusted source snapshotを再検証し、そのcommitをquiet-window-v14のsource authorityにする。
- quiet-window-v14とoperator-command-v9をfresh生成・封印してから、新しいauthorizationとしてactualを一回だけ実行する。
- actual-v8のresult/audit/runtime/capture rootsは変更も再利用もしない。
