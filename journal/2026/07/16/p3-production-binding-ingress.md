# P3 production binding ingress

## 前回の要点

session lifecycleとprofiler probeは実装済みだったが、production worker parserから
`InferenceRequest.aq4_p3_direct_trace`へbindingを渡す信頼済み経路がなく、監査はNO-GOだった。

## 今回の変更点

- manifest resident mode専用のbinding sidecar CLI path+SHA pairを追加した。通常JSONLは変更しない。
- 診断envとsidecarをstrictに同時要求し、legacy/benchmark wireでは診断envを拒否する。
- sidecarをcanonical/no-parent/no-symlink/single-link regular fileとしてfdからbounded readし、
  path/fd identityと全bytes SHAを再検証する。exact serde schema、1..64 entries、binding validation、
  request/binding ID uniquenessを必須にした。
- production parser後、SessionInferenceBackend前のresident wrapperがrequest IDを照合してbindingを
  一度だけ付与する。missing/exhausted/reuse/mismatch/unused shutdownをfail closedにする。
- M1/M8のCPU統合試験は実worker JSONL parserを通り、terminal observationがdiagnostic writerへ
  exactly oneで出る一方、stdoutには従来worker event以外が混入しないことを確認する。
- tamper、path escape、symlink、hardlink、duplicate/reuse、request mismatch、default-offを試験する。
- GPU、service、raw captureは実行しない。

## 次の行動

次の検証をjobs=1で完了した。

- `ullm-aq4-worker`: 23 passed
- `session_worker_backend`: 3 passed
- P3 Python suite: 135 passed
- workspace `--lib --bins --tests`: 1051 passed, 1 ignored

commit/tree/archive identityを固定し、Lunaの独立監査へ引き渡す。
