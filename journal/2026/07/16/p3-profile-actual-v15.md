# P3 profile actual v15 exact-one failure seal

## 前回の要点

- quiet-window v20 は 27 samples、353.686567482 秒、reset 0、最終確認通過で GO とした。quiet authority は commit `c8e223f1446e6cc5ab4c677e0cdf9ea8105b76a9`、tree `fb91daabecbb7e67c21231b746eb3d956535d523` である。
- operator-command v15 は commit `c76e46f06106db7489644493f2561b6dbec6b412`、tree `f46ed505c5dbd09fbbf43bf317cc8f3652581e7e`、command hash `520297d84df9f88eba8a98097222052079d8caccc15dbb74050dbbaaf93cc855` に固定した。
- micro-preflight では external SQ8 0、worker/owners の単独性、service epoch、fresh9 不在、quiet/operator の formal validation と clean diff を再確認した。

## 今回の変更点

- manifest の exact argv を `shell=False`、同一 PTY から exactly once 実行した。invocation は 1/1、retry/rerun は false、return code は 1、elapsed は 82,645,085,868 ns だった。exact-one authority は消費済みであり、この actual-v15 は再実行しない。
- finalizer は 1回だけ実行され、状態は `failed_immutable_evidence_preserved_restore_passed` になった。失敗時の6 evidence roots は内容を変更せず、root mode 0555、member mode 0444、各 `SHA256SUMS` 付きで封印した。
- 直接の失敗は `HarnessError: profile capture success artifact semantic binding differs` である。capture v2 producer が `profiler.target_environment.injected_fd_map_key = ULLM_AQ4_PINNED_FD_MAP` を記録した一方、maintenance validator は同 object を旧4キーの exact dict と比較しており、schema drift により binding が一致しなかった。
- workload 自体は status complete、resident model load 1、warmup 2、measured 10、transactions 12 を記録した。capture artifact は schema `ullm.aq4_p3_diagnostic_rocprof_capture.v2`、status `complete_diagnostic`、measurement/promotion eligible false、raw kernel rows 12,263、raw order inversions 248 である。row count、duration sum、correlation/dispatch の set と multiset は before/after で保存された。ROCTX marker は 12、measured run bindings は 10 だった。
- family classification の unknown/multi 集計と directional HIP/generic 集計は、artifact binding validation で停止したため後段の正式値を生成していない。raw trace と capture artifact は保存済みだが、これらを成功値として補完してはいない。
- restore は attempted/passed、duration 14,952,564,186 ns、poll count 6 で、6回目に readiness が通過した。新 epoch は service main PID 3268257、worker PID 3268350、active/running、NRestarts 0 である。KFD owner と amd-smi owner はともに worker PID 3268350 のみで、formal health endpoints はすべて HTTP 200、residual targeted processes は空だった。
- evidence commit は `99faf0066b93eb021fa83bea1b1a0193d9a79fd4`、tree は `0503c595c738ab66173918bd95986be613ddfc00` である。66/66 files について worktree、Git blob、`git archive` の SHA-256 が一致し、`validate-actual`、`validate-operator`、`validate-quiet` は postcommit でも通過した。

| root | files | Git tree | SHA256SUMS SHA-256 |
|---|---:|---|---|
| profile-maintenance-evidence-v12 | 5 | `c4cd3bfd028c2881071e9510d9da277c3e668fe3` | `263b3cd500152520e3d6066d3101f70e6ec2dee8f5dbeaf4fcc42404fc2a6225` |
| profile-execute-v11 | 7 | `f18fc17775531cb64612ddd1690371274e809a82` | `acd372452c80c334bd8534e19db1152a142219eb232d5a8958bbe0fe8e4f3eb9` |
| profile-execute-evidence-v11 | 8 | `b357627f2735ea2b122791f499f0ad1abd676e26` | `b843df5bc5d4a0c243632f1140f9da8ab32d1454e4dd7ecdc2e8cd9208970d78` |
| diagnostic-rocprof-capture-v11 | 40 | `8f41eb2f4aaf39c92c076073dc0f6458c14f8fc2` | `8ba65db03636c8167ee36d12f59b6cfc9de4e0aaa0df87cb11254ab1313994bb` |
| profile-operator-result-v15 | 4 | `019e81c43a6115d7f27f31a059a6bb93fc6b973f` | `4bb61f187baa4b6a001f5e6143e6c3d432b08a12e417f924cf19bd7b28ddcce9` |
| profile-actual-audit-v15 | 2 | `a3cf5553bcd3bb8c9de6a1b3bcd4b6a70b65e785` | `b930ba738496edda3aaad534127c9e7548a621e16de95a868b8d57f81b264fc4` |

## 次の行動

- capture producer と maintenance validator の `target_environment` schema を同じ authority に統一し、`injected_fd_map_key` の型・値・self-hash binding をテストで固定する。
- 修正後は今回の actual-v15 を再利用・再実行せず、新しい service epoch と新しい operator authority を作り、quiet と独立 GO 判定から改めて進める。
