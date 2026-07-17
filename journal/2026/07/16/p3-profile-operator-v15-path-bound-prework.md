# P3 profile operator v15 path-bound prework

## 前回の要点

- operator-v15骨格はquiet-v20、command/result/audit-v15として用意され、actual namespaceとcurrent authorityはUNBOUNDだった。
- previous-v14 actual failureとprevious-v13 blocked stateはhistorical専用validatorへ分離済みだった。

## 今回の変更点

- current profile actual pathsをmaintenance-evidence-v12、profile-execute-v11、profile-execute-evidence-v11、diagnostic-capture-v11へbindした。
- result-v15、audit-v15を含む6 rootはabsolute、unique、previous-v14の6 rootとdisjointである。
- fresh9はruntime、execute evidence、maintenance evidence、capture root、capture artifact、rocprof stdout/stderr、result、auditの正確な9 pathであり、現時点では全てabsentである。
- launcher-v11 execute-binding commit `9111b2a6c9479ebccb61a55641b5be52f86d5dda`、root tree `f76c878764aff5d4290bc48967928c0d1e1f6bac`をpath/namespace authorityとして追加した。
- launcher-v11は通常execute-v11 namespaceだけを証明し、profile namespace authority、execution authorityはいずれもfalseとして扱う。
- `CURRENT_V15_AUTHORITY_BOUND`はfalseのまま維持した。maintenance source、ready-v17、offline-v12、capture fixの最終authorityが揃うまでquiet/current audit/operator生成はfail-closedである。
- targeted testsは26 passed、source未commit時に意図的に失敗するsource last-change authority testを除く全体は60 passedだった。py_compileとdiff checkも通過した。
- source/testsは最終authority統合前のworking tree preworkであり、commitしていない。
- GPU、service、benchmark artifact、actual executionには触れていない。

## 次の行動

- maintenance source、ready-v17、offline-v12、capture fixのcommit/tree/blob/raw authorityが確定してからcurrent authorityを統合する。
- ready-v17のprofile outputが今回の4 pathと一致することを検証し、fresh9 absenceとprevious-v14/v13不変性を再確認する。
- 最終source/tests commit後にsource last-change authority testを含む全件を再実行する。
