# AQ4_0 runtime hardening activation execute r3

## 前回の要点

最初の activation は control route の readiness race により candidate live proof 前に rollback され、AQ4_0 closure 自体の欠陥ではないと診断済みだった。その後、`05014a8c`、`389a58f5`、`af7298ba` の control-route 修正で、bounded retry/backoff、stable worker identity、5 endpoint の coherent success、sanitized failure audit を持つ r3 plan を新規に seal した。worker、product、tokenizer、promotion source、candidate manifest は作り直していない。

## 今回の変更点

- 明示承認済みの plan `0e12fe09ad4d00578ee74f1bcc730a6b401e63a6fc91bb1d237346251e8f81f8` を、sealed `af7298bad50cfc7b8166c5505aaaffe0e9ad465f` control source で実行した。直前の default read-only preflight は `ready: true`、`blockers: []`、`production_activation_performed: false`、11 check PASS だった。
- 旧 active bytes と rollback copy はともに `5d015a013dcf70cea13dd9ed569d89ed2a025a17e14a6192ca18ee4cdadd1c8a`、candidate manifest は `c57a2b6c5827b8ddd102560b3f5efd879711705cf4d8a36f4d7872821d05fca4`、protected worker は 4,223,912 bytes / `1f93f21543af777adb0f00cc35d6857d0af432657ed74e7723636ace9dfca69b` で合致した。legacy live worker との `cmp` も byte-identical だった。
- candidate の guard contract は live と同順の unique 30 件を維持し、P3-only 6 flag の交差は 0 件だった。`llama-qwen35-udq4.service` は activation 前後とも inactive/disabled である。
- locked `--isolated-candidate-preflight` を再呼出しし、r3 plan に bind された immutable receipt を再検証した。no-replace receipt が既に存在する場合の正規動作は新しい candidate worker を二重起動せず receipt を再検証することである。receipt が記録する実 worker launch は `gfx1201` / `rdna4_aq4_resident`、ready 3,195 ms、SIGTERM cleanup `-15` で PASS だった。
- locked execute は 1 回だけ行い、immutable outcome `b022f91aa6118f379a79e59a6d35e30ba90b348511bdc789cfdd1c8c97f2d340` を `status: activated` で publication した。candidate reconcile と candidate live proof は PASS、rollback は不要で実行されていない。
- candidate live proof は 2 回の stable observation（1,139 ms）で、manifest/worker command/environment binding、worker PID `3757806`・PPID `3757204`・starttime `4461013`・executable hash、gateway health/ready/models と OpenWebUI health/models の全 5 endpoint 200、および model ID を確認した。
- OpenWebUI network namespace から one-shot OpenAI-compatible inference を実行した。HTTP 200、expected model ID、choice 1、nonempty assistant content、`finish_reason: stop`、usage 17/2/19 token で成功し、prompt/response 本文と credential は保存していない。
- service window は intentional stop/start 各 1 回、automatic restart 0 回。final `ullm-openai.service` は active/running、`Result=success`、StartLimit は 3 / 15 min のままである。postflight の AMD SMI card 2 は 34,208,743,424 bytes 中 7,426,916,352 bytes 使用で、candidate worker の VRAM 不足は発生していない。
- 新 active manifest は candidate hash `c57a2b6…fca4` となり、`worker.binary`、`product.root`、`tokenizer.root`、promotion receipt はすべて `/opt/ullm/aq4-runtime-hardening-v0.1/` の protected path を指す。worker hash は activation 前後で不変、rollback 用の旧 bytes `5d015a01…dadd1c8a` は別 immutable file として保全されている。

実行記録は `benchmarks/results/2026-07-26/aq4-runtime-hardening-activation-execute-r3/` に保存した。authoritative intent/outcome/proof は root-owned immutable records として `/opt/ullm/aq4-runtime-hardening-v0.1/activation-v0.2-r3/` にある。

## 次の行動

1. この r3 plan に対して activation を再実行しない。rollback/recovery も別の明示承認なしには実行しない。
2. Phase 7 の fresh campaign/browser/bundle v1 は今回の activation 承認の対象外である。必要なら hardened active manifest を起点とする別 task として実施する。
3. SQ8 campaign/authorization/candidate/release と `/opt/ullm/releases/` は今回変更していない。並行中の R9700 GPU F32 参照実装作業とは独立して扱う。
