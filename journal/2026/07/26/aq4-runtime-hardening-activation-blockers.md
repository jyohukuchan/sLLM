# AQ4_0 runtime hardening activation blockers implementation

## 前回の要点

- Phase 1–3 で AQ4 protected closure、promotion source、manifest-freezer control source は準備済みだったが、Phase 4 の fresh evidence/receipt/frozen candidate は未実施だった。
- AQ4_0 hardening activation には、AQ4→AQ4 専用の durable locked route と、post-hardening bundle v1 の immutable publication が未実装という2件の blocker が残っていた。
- live active manifest SHA-256 は `5d015a013dcf70cea13dd9ed569d89ed2a025a17e14a6192ca18ee4cdadd1c8a`、worker SHA-256 は `1f93f21543af777adb0f00cc35d6857d0af432657ed74e7723636ace9dfca69b` として保持する必要がある。

## 今回の変更点

- commit `d11085c4e119361cf0dca78e6cbe81cafcb9af6b` で AQ4 専用 control route を追加した。
  - plan schema は `ullm.aq4_runtime_hardening_activation_plan.v1`。
  - control source は同 commit の detached clean standalone clone を必須とし、tree と4つの route-tool hash を plan に固定する。
  - credential/source/runtime seal 完了後に immutable intent を `RENAME_NOREPLACE` で durable publication し、pinned active-parent dirfd 上の `renameat2(RENAME_EXCHANGE)` で candidate bytes だけを swap する。frozen candidate file は rename しない。
  - activation outcome は commit boundary であり、その後に fallible source check を置かない。SIGKILL/power-loss、`failed_restore`、`rollback_incomplete` は同じ lock と exact rollback bytes の recovery を持つ。失敗 recovery は unique audit だけを残し、success receipt pathname を消費しない。
  - default は read-only preflight。execute/rollback/recovery は exact plan SHA と literal confirmation を要求する。SQ8 final route、`llama-qwen35-udq4.service`、`gdm3` は参照しない。
- CPU/private-copy/mock tests は 58 件通過した。AQ4 固有の 13 tests で pre-intent drift、SIGKILL after intent、SIGKILL after swap、post-rename fault、live-proof failure restore、recovery retry audit、lock、stale SHA、one-shot receipt を確認した。GPU は使用していない。
- bundle v1 preparer を owner-bound `RENAME_NOREPLACE` + file/parent fsync + `0444`/nlink-one publication に変更した。production CLI default は `--required-uid 0`、validator は `--require-immutable-publication --required-uid 0` を提供する。
- read-only pre-plan preflight を実行し、`active.json`、systemd unit、gateway environment の SHA が計画値と一致することを確認した。evidence は次に保存した。
  - `benchmarks/results/2026-07-26/aq4-runtime-hardening-activation-v0.1/read-only-preplan.json`
  - `benchmarks/results/2026-07-26/aq4-runtime-hardening-activation-v0.1/read-only-preplan-control-source-pin.json`
- preflight は `ready: false`。frozen candidate、immutable rollback copy、activation plan、reviewed operations、credential seal set、`aq4-hardening-activation-d11085c4e119` sealed control-source clone が未作成である。active manifest、service unit、environment、SQ8 assets、GPU、service state は変更していない。

## 次の行動

1. `d11085c4e119361cf0dca78e6cbe81cafcb9af6b` の control source を `/opt/ullm/aq4-runtime-hardening-v0.1/control-source/aq4-hardening-activation-d11085c4e119/` に standalone/root-sealed clone として作成する。
2. Phase 4 の fresh AQ4 evidence/receipt/frozen candidate と exact immutable rollback copy、reviewed operations、credential seal list を作成する。GPU/service window が必要な evidence は別承認範囲で扱う。
3. complete plan の read-only preflight が `ready: true` を返しても、human operator が plan SHA/candidate SHA/rollback SHA/service window を確認して明示承認するまで `active.json` の swap は実行しない。
