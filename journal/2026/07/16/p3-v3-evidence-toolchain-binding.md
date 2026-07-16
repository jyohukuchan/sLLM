# P3 v3 evidence toolchain binding

## 前回の要点

- v2は、実行対象Candidate Aのcommit/tree/source archiveとP2 rejection chainを固定し、metric-free `no_eligible_candidate`を発行した。
- ただし、最終証拠を生成したtoolchain自身のcommit/tree/archiveと各tool hashを独立に固定していないため、completion auditではNO-GOとした。

## 今回の変更点

- `p3_implementation`をruntime identityとして維持し、別の`evidence_toolchain`に最終commit/tree/source archiveとqualification builder、raw builder、selector、finalizerのSHA-256を結合する。
- P2 rejection receiptからcomparison root、manifest SHA-256、SHA256SUMS SHA-256、row countを`p2_comparison`へexactに投影する。
- selectorはtoolchain archiveのpath/hash/size、現在実行している4 toolのhash、P2 comparison projectionを再検証する。selectionにもruntime、toolchain、P2 comparisonを別々に投影する。
- finalizerのexact inventoryをruntime archiveとtoolchain archiveの2本へ分離し、どちらもraw bindingと一致しなければ発行を拒否する。
- tool hash、toolchain archive hash/size、P2 manifest hash/row countの改ざんテストを追加した。P3関連Python 192 testsは通過した。
- v1とv2は履歴として保持し、再利用も削除もしない。v3は最終commit後にcreate-newで生成し、発行後はrepositoryを変更しない。

## 次の行動

- 最終commitとtreeを確定し、そのcommitのtoolchain archiveと固定済みruntime commit `94c8eae36a1cb4a21b44c006d474bda4d39189b5`のarchiveからv3を生成する。
- v3ではP2 comparison 24行、archive内の4 tool hash、commit/tree、SHA256SUMS、mode/nlink、activation不存在、runtime default OFFをread-onlyで監査する。
- 将来P2 official holdoutがpassedし、P3 promotion条件がすべて成立するまで、正規状態は`no_eligible_candidate` / default OFFのままとする。
