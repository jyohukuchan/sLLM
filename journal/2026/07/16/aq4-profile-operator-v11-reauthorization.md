# AQ4 profile operator v11 reauthorization

## 前回の要点

- operator-command-v10 は commit `d278a2ba71a0f30c56c7af8927990eb4d6ac1e26`、tree `5a4d1b0a3a0e30c4befaef2f6e2cf355b3af3484` に封印されている。
- v10 manifest の raw SHA-256 は `05f457d3cf17cc57db50add9456714407c2a442b94f9a3aa567e5d594cc64cff`、`SHA256SUMS` の SHA-256 は `7cd59f443e66667ba05fc7e1e2fb95326f8b60eda62ce2a3987d367bba8821c3` である。
- historical actual-v9 は `executed_sealed` として維持されている。

## 今回の変更点

- operator-command-v10 を immutable previous authorization として、sealed checksum、Git commit/tree/blob coverage、manifest self-hash、exact argv、maximum invocation 1、`shell=false`、fresh-output 再検査契約まで厳格に readback するようにした。
- v10 対象の runtime-v9、execute-evidence-v9、maintenance-v9、capture-v9 一式、operator-result-v10、actual-audit-v10 の9パスが全て不在である場合だけ、`authorized_not_invoked_preflight_blocked`、invocation 0/1、result/audit absent と記録する。
- v10 対象9パスの一部または全部が存在する場合は、partial output として fail-closed する回帰テストを追加した。
- v11 manifest validator は previous-v10 の commit/tree/hash、invocation 0/1、result/audit absent、fresh 9/9 absent、historical actual-v9 final-state を再検証する。自己ハッシュを再計算した状態改変も拒否する。
- current namespace を quiet-window-v16、operator-command-v11、operator-result-v11、actual-audit-v11 に更新した。profile-ready-v12、execute-binding-v9、runtime-v9、execute-evidence-v9、maintenance-v9、capture-v9 は不変である。
- v11 fresh output 9パスは全て不在であり、exact-one、rc 0、rc 17、historical actual-v9 final-state の既存テストを維持した。
- operator tests は17件全て passed、`py_compile` と `git diff --check` も passed した。GPU command、service操作、actual実行は行っていない。

## 次の行動

- この source commit を operator-v11 の trusted source authority として独立検証する。
- fresh quiet-window-v16 と operator-command-v11 を生成する前に、v11 fresh 9/9 absence と v10 の `authorized_not_invoked_preflight_blocked` を再確認する。
- actual は別の明示的 authorization と直前 preflight が揃うまで実行しない。
