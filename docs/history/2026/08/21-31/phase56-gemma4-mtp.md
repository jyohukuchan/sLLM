# Phase 56: Gemma 4 12B MTP assistant production path 履歴

> 状態: 完了
> 開始日: 2026-08-31
> 完了日: 2026-08-31

## 開始判断

WebUI／server統合とPhase 55 Gemma 4 MoEを完了した後、計画済みarchitectureのうち単一32 GiB GPUで公式artifactをactual検証できる
Gemma 4 MTPを次対象にした。MiniMax M3の公式BF16 repositoryは約854 GBであり、現行single-GPU production pathの実モデル検証には
適さない。一方、Gemma 4 12B assistantは845,719,296-byteで、既存の24 GB targetと同居できる見込みがある。

## 固定済み調査結果

- Hugging Face APIから公式assistant revision `46d4c6f13f0ac0ad827b915669b8df9b81c64c51`、gated=false、Apache-2.0、
  7 fileを確認した。
- exact revisionを`/home/homelab1/.cache/sllm/models/google--gemma-4-12B-it-assistant`へ取得した。model提供のcustom codeは
  import／実行していない。
- `model.safetensors`は845,719,296 bytes、SHA-256
  `3279c173daddd7186e79d652ad94022415736d3a1370625696c898429b06d6df`、header 5,360 bytes、48 BF16 tensorである。
- header SHA-256は`d0f1537ec1254122003a892254cefcf44c538f2cc42ba612b5791f4c6c5fdcb4`、catalog SHA-256は
  `fd87240fd7fe1beac3b7f39ff3d4ae93e4c5a3fb4192fc556a8a2f28d892cc3d`である。
- reader-only固定参照から、Q-only assistant attention、target KV read-only共有、一定draft position、3840+3840→1024 pre projection、
  1024→3840 feedback、assistant固有vocab headという意味契約を抽出した。詳細は
  [reader記録](../../../../references/gemma4-mtp-reader.md)を正本とする。
- assistantとtargetのtokenizerはbyte-identicalではなく、固定targetだけが`<|video|>` ID 258,884を名前付きtokenとして持つ。
  target tokenizerをwire正本とし、vocab幅と共通generation IDを検証するpair contractへ限定した。
- 公式最大context 262,144を32 GiBで保証せず、初期actual scopeを2,048とした。resident／workspace／KV admissionは実測し、
  モデル推奨contextと実行可能contextを混同しない。

## 実装結果

- assistantのfixed lock、48 tensor BF16 load plan、canonical `gemma4mtp` GGUF、derived lockを追加し、source／GGUFの全tensor byteを
  完全照合した。assistant GGUFは845,730,080 bytes、SHA-256
  `5fc3643bd68d460392e1caa6fd9df2cd4d862b7f49d62a09093b679a50cb3224`である。
- target embeddingとhidden rowを連結するpre projection、4層Q-only attention、target KV read-only view、post projection、assistant
  vocab headを既存semantic opへlowerした。assistant KVは0 byteで、target sliding layer 46／full layer 47だけを読む。
- model-neutral speculative transactionへdraft width 1を接続した。proposal中は公開target stateを更新せず、accept／reject／length／stop／
  cancelで未消費target rowをrollbackまたは破棄する。絶対RoPE positionと動的長さの再bindも同じrequest ownerへ固定した。
- 通常`generate`、direct `benchmark`、`sllm-server` static起動、OpenAI Chat Completions非stream／SSE、raw Completions、metrics、
  dynamic model library、WebUI load／unloadへ統合した。static serverは`--mtp-assistant-gguf`と
  `--mtp-assistant-derived-lock`を`--draft mtp-auto`と組み合わせる。WebUI model libraryはassistantを単独load不可のcompanionとして表示し、
  exact `gfx1201`かつtarget＋assistantがVRAM内のときだけtargetへ結合する。
- quantized Gemma targetのdynamic preflightがBF16論理展開量を宣言してload後に隔離される既存問題を修正した。FP8／NVFP4／BF16のpacked
  payload、scale、alignment、graph constantから9,201,218,276 bytesをhost-onlyで算出し、plan digestは維持する。paired targetの宣言量は
  assistantを加えた10,046,932,204 bytesとなる。dynamic metricsもloaded lifecycleのbackend snapshotを参照し、resident memoryを0ではなく
  同じ10,046,932,204 bytesとして公開する。

## exact `gfx1201`実機結果

- 最終release binary SHA-256は`sllm`
  `37368b55f9abe886e2316342e2b7e40e2c6f5e2ad4b24be516f2059338b3a6b4`、`sllm-server`
  `0c9a99dfc233ccb7a66110f035b076876065be3e24ee3e05091227a6827fd4a1`である。R9700 UUID
  `GPU-a8e9ddefa2d60f55`だけをvisibleにし、logical device 0、exact `gfx1201`、context 2,048で実行した。
- fixed input `[818,5279,529,6056,563]`／4 outputでtarget-onlyとMTPはgenerated／visible
  `[236772,236770,236770,236772]`、raw text `-11-`、length、usageを完全一致した。全sampleはHIP-only、fallback 0、nonfinite 0、
  retryable cleanup 0、durable quarantine 0である。
- final 1 warmup／3 measured direct benchmarkのtarget-only prefillは`13.733350`、`13.800815`、`13.759520` tok/s、decodeは
  `9.332870`、`9.344950`、`9.347957` tok/sだった。residentは9,201,218,276 bytes、runtime peakは9,980,138,784 bytesである。
- MTP prefillは`13.725301`、`13.705836`、`13.712376` tok/s、decodeは`2.934036`、`2.902031`、`2.899310` tok/sだった。
  residentは10,046,932,204 bytes、runtime peakは11,574,306,638 bytesである。4 sample合計で12 proposal／0 accept／12 rejectとなり、
  この短いcaseではtarget-onlyより遅い。性能採用条件ではなく、幅拡張・融合は別profile作業とする。
- static serverでhealth／ready／models、Chat非stream、Unicode SSE、raw Completions、code、stop除外、連続要求、client disconnect、
  recovery、metricsをPASSした。shutdown auditは全requestでHIP-only、fallback false、MTP reject accounting、request/workspace cleanup 0、
  final current bytes 0を記録した。
- model sourceなしの標準起動でAPI `127.0.0.1:18080`とWebUI `localhost:65458`を同時起動し、runtime URL、R9700認識、folder選択、
  companion表示、dynamic load、raw／SSE生成、resident metrics、unloadをPASSした。終了は`clean=true`、両port閉鎖、WebUI子process 0、
  GPU process 0、VRAM 59,912,192-byte基準値復帰を確認した。

## 検証とreview

- workspace全target test、workspace全target clippy `-D warnings`、rustfmt、`git diff --check`をPASSした。WebUIは12 test、typecheck、lint、
  format check、production buildをPASSした。未追跡9.3 GB targetを必要とするpacked resident回帰も明示`--ignored`実行で
  9,201,218,276 bytesを確認した。
- 1回のintegration reviewでpaired resident accounting、exact target／VRAM admission、dense Gemma benchmark不足、static server flag不足、
  target量子化表記の誤りを検出し、各findingをfocusedに修正・再確認した。性能向上を正しさの完了条件へ追加していない。

[対応する保存済み計画](../../../../plans/archive/2026/08/21-31/phase56-gemma4-mtp.md)
