# AQ4 P2 holdout plan-swap hardening

## 前回の要点

- outer `_execute`はpreflightを読んでrescueを予約した後、inner `_execute_inner`が同じpathを再読していた。
- 2回のread間でrename replacementされると、rescueとattempt/commandが別planから導出される余地が独立監査の唯一blockerだった。

## 今回の変更点

- preflightを`O_NOFOLLOW` stable fdから一度だけbounded read/hash/parseし、fdをspawn境界まで保持するsingle-plan contextへ変更した。
- command、environment、attempt/result/rescue/stdout/stderr pathを同じparsed planから一度だけ導出し、innerからpreflight pathとargsを除いた。
- attempt前、marker後、spawn直前に同fdのpre/post `fstat`とpath `lstat`を初期fingerprintへ照合し、inode/mode/nlink/size/mtime/ctime driftを拒否する。
- two-read swap、old/new attempt/rescue分離、marker後replacement/ctime drift、symlink/hardlink、escaped exception、success/failure/retryのtestsを追加した。
- GPU、service、sudo操作は行っていない。

## 検証

- holdout runner focused: 28 passed。
- holdout関連Python: 40 passed。
- `cargo check -p ullm-engine --bin ullm-aq4-fidelity-capture`: pass。
- `cargo test -p ullm-engine --bin ullm-aq4-fidelity-capture -- --test-threads=1`: 9 passed。
- `uvx ruff check`、`uvx ruff format --check`、`py_compile`、`git diff --check`: pass。

## 次の行動

1. commitを固定し、新source全体を独立監査へ渡す。
2. GOまではholdout executeを禁止する。
