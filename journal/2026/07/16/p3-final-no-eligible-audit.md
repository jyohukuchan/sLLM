# P3 final no-eligible audit

## 前回の要点

- P2 calibrationは事前宣言したrelative-L2拒否条件に触れ、holdout未実行の`rejected_no_go`で終端した。
- P3 production条件の7 prompts×10 measured runs、M=128+other、paired full-model CIはpromotion rawにだけ適用される。
- Candidate A実装はcommit `94c8eae36a1cb4a21b44c006d474bda4d39189b5` / tree `88f2c0e451430b1305b181e9403caabede6f1107`に固定したが、promotion build/profileは行っていない。

## 今回の変更点

- 実P2 rejectionからmetric-free `qualification_only_diagnostic` rawを生成し、canonical `no_eligible_candidate` selectionを固定した。
- immutable packageを`/tmp/ullm-p3-no-eligible-evidence-94c8eae3-9e619f85-v2`へcreate-new発行した。
- packageはP2 rejection receipt `9e619f85…`、P2 SUMS `ce93213d…`、policy `302c3219…`、plan `3f93c95b…`、actual receipt `bc13dcf6…`、qualification self-hash `2783fb1c…`、raw semantic hash `5e621285…`、selection file hash `10f900ff…`を結合する。
- package SHA256SUMS hashは`748cda7fad3db4f4b75916fd4b3d2f25b8a7f9d42601c27b2f542a77d0930de2`である。directoryは0555、全fileは0444/nlink1で、activation artifactは存在しない。
- direct-route環境変数だけでworkerを起動するとexit 1になり、runtime default OFFを実行確認した。
- 32 MiB JSON readerで844 MiB archiveを扱おうとして停止したv1は失敗証跡として保持した。raw/selection/SHA256SUMSは発行されていない。

### 項目別監査

- P3測定条件: promotion candidateにのみ必須。qualification-only variantは性能fieldを持てず、条件を迂回しない。
- P2 fidelity: rejected、holdout `not_started`、残り1、実行なし。GOへ変換していない。
- Candidate A build/profile: `not_built_for_promotion` / `not_measured`。perf、CI、VRAM、fidelity値を記録していない。
- selector: 全候補ineligible、selected candidateはnull、statusは`no_eligible_candidate`。
- production activation: artifact不存在。runtimeはdefault OFF。
- 改ざん耐性: cross-variant、extra metrics、fake CI、hash/size/type swap、receipt swap、symlink、hardlink、replace/truncate/grow、並行publishをfail-closedにした。
- OOM対策: archiveは1 MiB固定chunkでstreaming SHA-256し、全体をJSONまたはメモリへ保持しない。
- 検証: P3 Python 187 tests passed。`cargo test --workspace --lib --bins --tests -j 1` passed。
- 全workspace examples込み実行は、既存のmodule断片`examples/sq8_ck_serving_performance.rs`をstandalone exampleとしてcompileする既知構造で失敗した。P3変更に起因せず、指定されたlib/bin/integration test範囲は成功した。

## 次の行動

- 現時点の正規完了は`no_eligible_candidate` / default OFFである。
- 将来P2 official holdoutがpassedし、P3 promotion測定が全条件を満たした場合だけ、新しいqualified raw、selected artifact、production activationを別attemptとして発行する。
