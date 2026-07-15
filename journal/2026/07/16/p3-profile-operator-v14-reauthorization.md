# P3 profile operator v14 reauthorization

## 前回の要点

- quiet-v18とcommand-v13は封印済みだったが、封印後かつactual invocation前に外部cargo/SQ8 familyが再出現した。
- command-v13は実行されておらず、invocation 0/1、fresh outputs 9/9 absent、result-v13/audit-v13 absentだった。
- result/audit schemaはactual invocation 1と実maintenance/restore evidenceを必須にするため、preflight abort文書をresult-v13/audit-v13として生成しない方針にした。

## 今回の変更点

- current namespaceをquiet-window-v19、operator-command-v14、operator-result-v14、actual-audit-v14へ更新した。
- ready-v16、offline-reassembly-v11、maintenance-evidence-v11、profile runtime/execute-evidence/capture-v10は維持した。
- previous command-v13専用のpathとauthority constantsを追加した。current v14 output constantsをprevious-v13 validatorから参照していない。
- command-v13をcommit `764045355ee06c3b5c53f296d4bcbe47e1495ece`、overall tree `cb73e9c7c34c884eac567510f6d89da238b57a49`、root tree `d187b2902aa9f83503c17d6c0c8665210744f2e0`としてGit objectから検証する。
- command-v13 manifest raw SHA-256 `78168089ff34e2eb8560bcaa85c94f49c0f3ae23ee4a614f0d0fc7e077a0d4f0`、selfhash `42c8498adc6c8f97382ef17421d3145a14d50126a549a66d0693f114f8cad313`、SUMS SHA-256 `1c157f9d864b4e75d62e2acc7b5b5189b1765e3795b3109ef4e815df26b87fd6`、command SHA-256 `5693d75b17f91187b6841566815ad717d001a91280d651860aa127dc20277079`を固定した。
- previous-v13 stateは`authorized_not_invoked_preflight_blocked`、reason `external_owner_after_seal_before_invocation`、invocation 0/1、result/audit absent、fresh9 absent、actual/GPU/service touch falseとして返す。
- reasonはv13 manifestに記録済みの過去イベント証拠ではなく、v14再認可manifestが封印するpolicy classificationである。
- quiet-v18はprevious authorityとしてexact commit/tree/root/hash/policyを検証する一方、current quiet-v19としての再利用をschema違反で拒否する。
- current v14 fresh outputsは9件のcount/absentだけでなく、exact path listとの完全一致を検証する。
- v14のstatic manifest validatorはprevious-v13 live absenceを再評価しない。共有actual outputが将来生成された後も、v14 manifest内に封印したpre-execution snapshotを検証できる。
- v14 actual validatorはmanifest commit、manifest file SHA、semantic SHA、command SHAをoperator resultとactual auditへ相互bindingする。

## 検証

- source/tests commit: `c69070e12c474c62f26c671dee5bf1c2ea72d570`
- commit tree: `a35bc99cc40eaaf4883d31cc5f5d0a10dbbd295e`
- operator source blob: `552b79a8fdc0303f8707bf571b4e55b99742d49c`
- operator source SHA-256: `b5cb1e74b95e11815a67388fda88e4eff48f46859735663a6f10d8a1f9dbc3ab`
- test blob: `7b1715ef0eb36b8b5fec801f8def6aff1f4bac33`
- test SHA-256: `aa8813d4792299c8be43146271bf7218f7f8ee17327695ba729adc83b8b72769`
- operator tests: `56 passed in 4.99s`
- `py_compile`、`git diff --check`: 成功
- 9個のprevious-v13 fresh pathを各1個ずつ存在させるpartial-state testは全てfail-closeした。
- 実`audit-current`: clean。production worker PID `2357251`だけがAMD-SMI/KFD ownerで、service MainPID `2356631`はactive/running、NRestarts 0だった。
- v14 fresh outputs: 9/9 absent。
- quiet-v19、command-v14、result-v14、audit-v14: absent。
- actual execution、GPU workload、service操作、artifact生成: 0回。

## 次の行動

- source/tests authorityとv13 reauthorization stateを独立監査する。
- 外部family不在とproduction owner単独状態を再確認してから、quiet-v19を既定の完全窓で収集する。
- quiet-v19のGO封印後にだけcommand-v14をexact-one pendingとして生成する。actualは別の明示的指示まで禁止する。
