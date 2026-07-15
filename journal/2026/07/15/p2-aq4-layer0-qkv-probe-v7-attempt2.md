# AQ4 layer-0 QKV standalone GPU probe attempt2

## 前回の要点

- Diagnostic loader は production batch-plan を変更せず、standalone single-matvec probe だけを診断経路として分離した。
- Build receipt と binary は commit `4a4b0e28eb27fa6710a339e470ee80d21d602680` に固定し、cold/cached build の receipt SHA は一致している。
- attempt1 は `Aq4MatvecBatch` の production admission failure で停止し、GPU output は生成されなかった。

## 今回の変更点

- main HEAD `5e7ecce84c54288453cf102405d6eaf845e6d501` で、固定 gate の preflight を実施した。input/package/active/profile/worker、probe binary、build receipt の pinned SHA、clean BASE、service `active/running`、runtime directory/lock owner を確認した。
- sudo credential 不在で最初の execute 呼び出しは service access 前に終了した。この呼び出しでは service stop と GPU 実行は発生していない。
- sudo を同じ PTY で確立した後、execute gate を1回だけ実行した。GPU probe の再実行はしていない。
- 結果は `status=valid`、`classification=unclassified`、`promotion_eligible=false`、`fused=false`。R9700 は HIP logical device 1 / HIP ordinal 0 / `gfx1201` として実行され、3 rows / 98304 bytes を生成した。
- output SHA は `24248fd1c4b4b7186f9b048a7fa4c69925904a04b265a273390089df7312545e`、report SHA は `4cce8b4a55c506d94801201314237cab1fc0adaaa70e27861116ad15e1f7efc1`。
- observer は2サンプルを取得し、`observer-failed.marker` は生成されなかった。raw monitor SHA は `0ed22b60c354439ecab6340177295fec064bd9f2385b0af816652ee9b4757d35`。
- cleanup 後の service は `active/running`、MainPID `1722227`、runtime directory `mode750 uid1000 gid1000 nlink2`、lock `mode600 uid1000 gid1000 nlink1`、lock owner `ullm-openai-gat` だった。active/package/worker SHA は gate pin と一致した。
- cleanup の service start 直後に `/run/ullm` の mount namespace が一度失敗し、systemd の `NRestarts` が1になった。その後の自動 restart で service は active/running に復旧した。この transient restart は attempt2 state と本 journal に記録している。
- raw output/report/observer/markers/state と `SHA256SUMS` を `benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-layer0-qkv-gpu-probe-v0.1/attempts/attempt2/` に保存した。archive `SHA256SUMS` は `b353bf460d21b91998f595e43126069663f014fb2e7737b1623e824388329a00` で、9/9 検証に成功した。
- 実行後 BASE は `attempts/` と `input/` だけに戻した。判定は threshold-free diagnostic のため promotion には使わない。holdout は未観測である。

## 次の行動

- attempt2 archive と本 journal を通常の限定 commit として保存する。
- GPU/service は再実行せず、CPU output との比較や以降の統合判断は別担当に委ねる。
