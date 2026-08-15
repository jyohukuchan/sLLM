# Phase 17 Qwen3.5 MTP・vision履歴

## 2026-08-16: 詳細計画作成

- fixed `Qwen/Qwen3.5-4B` revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`のknown-unconsumed
  MTP 15 tensorとvision 297 tensorを正式消費する計画を作成した。
- debugging範囲を分けるため、text-only MTPのreader/oracle/graph/speculative transaction/serviceを先に完了し、
  image processor/vision graph/multimodal prompt/APIを後から実装する順序にした。
- MTPはgreedy token完全一致、stochastic rejection/residual sampling、accepted prefixだけのopaque KV publication、
  stop/EOS/cancel、target別内部provider選択を受入条件にした。Phase 16のFP8 KVを代表caseで回帰する。
- vision processorは同じrevisionのpixel area `65,536..=16,777,216`、patch 16、temporal patch 2、merge 2、
  mean/std 0.5とspecial tokenを固定した。NumPy/Pillow oracleを使い、PyTorchは使用しない。
- OpenAI公式OpenAPI 2.3.0とImages/vision guideを2026-08-16に確認した。Chat Completionsのtext/image content arrayを
  versioned profileへ追加するが、初期server sourceはBase64 data URLだけとし、HTTP(S) fetch/Files APIを実装しない。
- MTPとvisionを個別にPASSした後だけcombined image+MTP smokeを行う。本時点ではsource、model lock status、API、fixtureを変更していない。

## 2026-08-16: MTP component完了

- fixed lockからMTP 15 tensorをexact manifestへ昇格し、shared embedding/output、shape/dtype/range、1-layer topologyをload前に検査した。
- MTP graphはtarget hidden rowとcandidate embeddingをnorm/fc/decoder/logitsへ通す。model-neutral verifierはgreedy完全一致と
  stochastic rejection/residual sampling、RNG順、stale generation、abort/commitをhost boundary testで固定した。
- real-weight evidenceは両targetでdraft `[198,248044]`、target verify `[198,248045,248045]`、accepted draft 1、
  emitted `[198,248045]`、deterministic replay、HIP-only、fallbackなし、cleanup 0だった。R9700最終runはtarget 1968 dispatch、
  MTP 125 dispatch、10.879秒だった。
- 現行verifyはdraft分のtarget forwardを逐次実行してtarget-onlyのforward数を減らさない。従ってcanonical V620/R9700の通常CLI/APIは
  target-onlyを内部選択し、MTP許可flagや品質警告を追加しない。将来batched verifyで利益が出るtargetだけ採用を再評価する。

## 2026-08-16: vision/multimodal完了

- PNG/JPEG/WebP/non-animated GIFのmagic/decode検査、32 MiB encoded上限、pixel/aspect/image count/visual token境界、EXIF無視、
  Catmull-Rom resize、RGB normalize、temporal duplicate、patch/merge順をRust frontendへ追加した。
- 独立NumPy oracleの256x256 fixture digest
  `f1e51663a9ea2832a67e5157ca11bc42206aaf186897866dab8c779d08ee3a2e`とRust outputが一致した。
- vision 297 tensorをlazy residentへuploadし、patch projection、position、24 transformer block、merger/projectorを実装した。
  dense matmulは既存HIP semantic op、vision固有の小さいnorm/bias/GELU/attention/position transformはdeterministic host処理である。
- typed multimodal promptはlocked image-pad runだけをprojected BF16 rowへ置換し、3-axis mRoPEをnative attention preprocessへ渡す。
  text-only requestはvision residentを確保せず、image encodeはprefill前に一度だけ実行する。
- Chat Completionsはuser content arrayのordered text/data-URL imageをstrictに受け、remote URL、Files ID、unknown part/detail、
  non-user imageを拒否する。CLIは`generate --image PATH`を最大2回受け、最終user textの前へ画像を置く。
- V620最終evidenceは256 patch/64 visual token、vision 198 dispatch（deterministic 2回合計）、text 986 dispatch、
  projected digest `238f08aa155244f913d1701b03a96ca0eb6bcdb9b6b147e242b86b8222907653`のreplay一致、
  prefill token 198、decode token 248045、20.278秒だった。
- R9700最終evidenceは同じgeometry/dispatch、projected digest
  `e8ddb0dd639f652b4b8569980dad6588cd6e431c926c56b31185305bae31fd87`、prefill token 198、decode token 248045、
  19.620秒、全HIP、fallbackなし、cleanup 0だった。実CLI local PNGも77 prompt tokenから`The`（token 760）を生成し、
  493 kernel dispatch、fallbackなし、cleanup 0でPASSした。
- R9700の実serverでも同じPNGをOpenAI non-stream/SSEへ送り、両方`The`、usage 79+1、SSE terminal chunkと`[DONE]`を
  PASSした。lazy vision residentを含む最終shutdownはmodel-ready 9,078,620,672 byteからfinal current/request/workspace 0、
  retryable/durable cleanup 0へ到達した。
- vision weight量子化、remote fetch、cross-request image cache、videoは非対象のまま。次はPhase 18 MoEとする。

[対応する計画](../../../../plans/archive/2026/08/11-20/phase17-qwen35-mtp-vision.md)
