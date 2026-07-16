# P3 profile operator-v16 authority integration

## 前回の要点

- operator-v15 は、command commit `c76e46f06106db7489644493f2561b6dbec6b412` に基づいて 1 回だけ実行された。
- actual commit は `99faf0066b93eb021fa83bea1b1a0193d9a79fd4`、root tree は `0503c595c738ab66173918bd95986be613ddfc00`、6 root・66 file である。
- capture artifact は schema v2 の `complete_diagnostic` まで成功したが、maintenance が `HarnessError: profile capture success artifact semantic binding differs` で拒否した。return code は 1、retry は未実施、outer restore は成功した。

## 今回の変更点

- current namespace を ready-v18、quiet-v21、operator command/result/audit-v16、maintenance-evidence-v13、runtime/execute-evidence/capture-v12、offline-reassembly-v13 に更新した。current 6 root と previous actual-v15 6 root は非重複である。
- ready-v18、offline-v13、maintenance fix、launcher/execute-binding-v12 の sealed authority を Git と SHA256 から再計算し、`CURRENT_V16_AUTHORITY_BOUND = True` とした。
- previous operator-v13、actual-v14 の履歴 validator を維持し、quiet-v20、operator-v15、actual-v15 の commit/tree/inventory/semantic failure/finalizer/restore poststate validator を追加した。
- previous actual-v15 の capture success schema v2 と maintenance semantic rejection を別々の事実として固定し、Git 上の 6 root・66 file coverage を検証した。
- unbound override 時には prepare、current audit、finalize、validate が artifact を読む前に停止する test を維持した。
- authority loader の coverage は ready-v18 が 4 files、execute-binding-v12 が 3 files、offline-v13 が 42 files である。offline-v13 の source actual seal は 66 members である。
- operator source/test commit は `4869fde48ca872da70b09b029ebdd9da169fc4b1`、tree は `d531b2e1bfb27e959ff7f8eaa6dc6259a65efcd9` である。source blob/raw SHA256 は `2ccdcb1e5e0e7725c6b965ac0cd86b91d44f9920` / `2990ee98bd24a41c536725a3edd977e6248442b79fc181e342326dd0be9a51c1`、test blob/raw SHA256 は `286215fac77bfb19cffe95d98e1e6a60421082d5` / `d1882f7328781fd8957c86174a7b06b95f35c9cff8c3ca8102cd8465a6d76840` である。
- `python3.12 -m pytest -q tests/test_prepare_aq4_p3_profile_operator.py` は 75 tests passed、`py_compile` と `git diff --check` も通過した。
- read-only `audit-current` は `status=clean`、fresh 9/9 absent、targeted processes 0、`service_touched=false`、`actual_executed=false` である。formal health SHA256 は `b032d38fcdb8e17f2452daa47ce07f2335875451df2ac47f73d117a4331b3722` である。

## 次の行動

- 独立 QA では source commit/tree/blob/raw SHA256、ready 4 files、execute-binding 3 files、offline 42 files、previous actual-v15 66 files、fresh 9/9 absent を再確認する。
- quiet-v21 と operator command-v16 の生成は、明示的な次工程まで行わない。
- actual、GPU command、service operation は実行しない。
