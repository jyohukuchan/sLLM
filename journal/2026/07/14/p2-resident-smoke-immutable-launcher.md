# P2 resident smoke immutable launcher

## 前回の要点

R final runnerとB sidecarは固定済みだが、root/B/R/validatorを一括で選び、validatorからrunnerの順序を強制するLは未実装だった。

## 今回の変更点

- input root exact19、B manifest、e906592 runner、2e39b78 validator、319d618 resident binary、served manifest、device 1、lock path、one-case ID/hashをlauncher constantsへ固定した。
- 全pathのabsolute/no ancestor symlink、regular file、nlink=1、SHA-256、open前後identityを検査した。
- input rootとB sidecarは初期snapshotを取り、validator後とrunner後にdirectory/member identityを再検証した。
- launcher validator subprocess 1回が成功した後だけrunner subprocess 1回を起動した。R内部mandatory validator 1回とfake-ready child 1回は別countとして記録した。
- runner planはB plan SHA-256とexact一致し、1 case、12 transactions、warmup 2、measured 10、smoke-only、promotion falseを確認した。
- stdout/stderr/exit/report/plan/result/self SHAとprocess countsをatomic no-replace evidenceへ保存した。
- unknown/duplicate/rebind、ancestor symlink、validator failure時runner 0、runner failure single attempt、late replacement、existing output、execute拒否の負例を追加した。
- actual executionは拒否し、GPU command、model load、service変更・停止は実施していない。
- canonical dry-run evidenceを`resident-one-case-smoke-launcher-dry-run-v1`へ保存した。launcher self SHA-256は`8cd38aabc60eba5dfdcc3adc46421cbe7508bfd95bcb1d8b56b410f1a0f1fa81`、result/B plan SHA-256は`bc449219b5e32882d1bca4663abf1eac631dd59c5d0503f1ee287d76ebeabd9c`である。
- launcher testsは7件通過し、canonical evidenceの`SHA256SUMS`は全memberで通過した。

## 次の行動

actual smokeには、driver optionを安全に透過できるR CLIとexecute用L trust rootが必要である。device lock/cleanup evidenceを追加後、別の承認単位で実行する。
