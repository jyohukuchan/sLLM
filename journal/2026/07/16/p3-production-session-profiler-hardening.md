# P3 production session / profiler hardening

## 前回の要点

candidate-A direct traceにはrequest単位collectorとruntime/profiler assemblerがあったが、production
session/worker lifecycleとの接続、M=1 route accounting、profiler executableの再実行検証が不足していた。

## 今回の変更点

- 明示的なAQ4 P3 binding metadataを通常validation後、最初のdispatch前に一度だけ開始する。
- M=1とM>1を同じcollectorへ記録し、finish/error/cancel/resetを同期境界後に一度だけ確定する。
- terminal observationを最大64 KiBのstderr JSON 1行へ限定し、worker wireを変更しない。
- profilerの検証済みbytesをsealed memfdから`--version`再実行し、timeout、exit、stdout/stderr policy、
  SHA-256 receipt、stored versionをproducerとassemblerで再検証する。
- CPU試験でM=1/M=8、通常終了、dispatch error、cancel、reset failure、default-off、binding mismatch、
  resealed probe改ざん、executable置換、timeout/stderr拒否を確認する。GPU/service/raw captureは行わない。

## 次の行動

jobs=1検証結果はP3 Python 135 passed、workspace Rust lib/bin/testは1044 passed・1 ignoredだった。
`cargo test --workspace`だけは既存SQ8 example moduleをstandalone exampleとしてcompileするため失敗したが、
SQ8を変更せず、exampleを除く全workspace CPU suiteで回帰がないことを確認した。次はcommit identityを固定する。
