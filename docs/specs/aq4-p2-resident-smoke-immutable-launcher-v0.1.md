# AQ4 P2 resident smoke immutable launcher v0.1

## 前回の要点

Rはgeneric one-case runner、Bは791a20c input rootを変更せずにR/validatorの実dry-runを固定するsidecarである。B自身は`launch_eligible=false`であり、validatorとrunnerの選択・順序を強制するLが未実装だった。

## 今回の変更点

Lは次のtrust rootをsource constantsとして固定する。

- input rootのabsolute path、directory device/inode、exact 19 membersと各SHA-256、aggregate fingerprint
- B sidecarのabsolute path、exact member set、manifest SHA-256
- R commit `e906592`、tree、Git blob、source SHA-256、sidecar内single-link runner path
- checked-in validator commit `2e39b78`、tree、Git blob、source SHA-256、absolute non-symlink path
- normative resident commit `319d618`、detached binary SHA-256 `62f720…30f1`、served manifest SHA-256、device index 1、lock path
- exact one-case IDとruntime-bound case SHA-256
- fixed Python interpreter pathとSHA-256

全trust pathはabsolute/no-parent-traversalで、全ancestorを`lstat`してsymlinkを拒否する。固定fileはsingle-link regular fileとしてopen前後のidentityとSHA-256を確認し、input/B directoryと全memberはvalidator前、validator後、runner後に同一identityであることを確認する。

dry-runではLがchecked-in validatorをsubprocessとしてexactly once実行し、成功reportがroot/B/status/promotionに一致した後だけRをexactly once起動する。R内のmandatory validatorとsynthetic fake-ready childは別countとして各1回である。Lが起動する順序は`validator → runner`で固定し、validator失敗時はrunner count 0とする。

成功planはB plan SHA-256とexact一致し、1 case、12 transactions、warmup 2、measured 10、`smoke_only=true`、`promotion_eligible=false`を要求する。subprocess argv、exit、stdout/stderr file SHA、validator report SHA、plan/result SHA、process counts、実行時launcher self SHAをevidenceへ保存する。launcherは自分自身のSHAをconstantsへ埋め込まないため、self-hash cycleはない。

evidence outputは新規absolute directoryだけを受け入れる。各fileはtemporary single-link regular fileを作成し、`fsync`後にhard-link no-replaceで公開する。既存outputへの上書きは拒否する。失敗時も開始済みprocessのstdout/stderr、exit、stage、runner開始有無を保存する。

今回の承認範囲はdry-runのみである。`execute` modeはsubprocess開始前に拒否し、GPU command、model load、service変更・停止を行わない。resident driver optionをRの現行`nargs=+`へ安全に透過できない問題もあるため、actual実行は別のR/L更新と明示承認が必要である。

## 次の行動

actual one-case smokeへ進む場合は、resident driver argvを曖昧なく渡せるR CLI、Lのexecute専用authorization boundary、device lock evidence、終了時lock cleanup検証を追加する。その新しいLを別trust rootとして固定し、GPU実行を改めて承認する。
