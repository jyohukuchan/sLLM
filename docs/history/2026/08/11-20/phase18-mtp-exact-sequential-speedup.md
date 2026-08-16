# Phase 18 MTP逐次承認・target-only数値同一・最低限の高速化 履歴

## 2026-08-16: 詳細計画作成

- ユーザー指示により、従来Phase 18だったMoEをPhase 19へ繰り下げ、Phase 17で未完了だったMTP production性能統合を
  Phase 18へ割り当てた。
- llama.cpp issue [#25618](https://github.com/ggml-org/llama.cpp/issues/25618)の量子化target分岐を踏まえ、llama.cpp MTP実装の
  一括copy/adapt/portを行わず、issueはdefect classと回帰matrixのsourceに限定した。
- 通常のM=1逐次target decodeを数値oracleとし、draftの逐次承認、最初のrejectでの打切り、accepted prefixだけのKV/state commit、
  greedyおよび同seed stochasticのtarget logits/token/RNG/visible output一致を受入条件にした。
- 高速化はM=1と同じrow reduction/roundingを保つserial-equivalent batch、device-side orchestration、prepared segment、MTP overhead削減で
  行う。異なるM>1数値pathを許容差で通さない。
- 既存の性能candidate方針に従い固定3% floorを置かず、target/case固有noise envelopeを越えるpaired MTP off/on改善を採用条件にした。
  Phase完了には少なくとも一つのcanonical targetでfull-generation改善とMTP auto-selectionを必要とし、改善しないtargetは通常UXを
  変えずtarget-onlyを内部選択する。
- MoEをPhase 19、GGUF統一と残機能をPhase 20、人間によるREADME整備・発表をPhase 21へ繰り下げた。

## 2026-08-16: serial-equivalent verifyと逐次commit

- BF16 MatmulへM=2..8専用のserial-row providerを追加した。複数rowでweightを共有するが、各rowのdot productは通常M=1と
  同じlane分割、reduction、roundingを使う。一般M>1 hipBLAS pathの数値差を許容差やtoken一致で採用していない。
- target decode blockはpending blockとして実行し、先頭から逐次承認したinput rowだけをcommitする。途中rejectではKVと
  linear-attention stateをpre-block世代へrewindし、accepted prefixだけを同じtarget pathでreplayする。MTP側もdiscard数だけ
  opaque transitionをrewindし、全accept時だけbonus位置までcatch-upする。
- generation serviceの既存one-token sampler/stop/usage loopの前へ内部adapterを置いた。greedyはaccepted canonical target stepだけを
  queueし、sampled requestは同じpublic RNGとlogits順を保つtarget-only内部providerを自動選択する。CLI/APIへMTP flag、opt-in、品質警告、
  起動コマンド差は追加していない。
- production auto-selectionはfixed Qwen3.5-4B、BF16、text-only、greedy、exact `gfx1201`の検証済みrowだけdraft width 1を選ぶ。
  `gfx1030`、FP8/NVFP4、vision、sampled request、未計測tupleは同じユーザー操作のままtarget-onlyを選ぶ。

## 2026-08-16: exact GPU evidence

- fixed lock fingerprint `sha256:f143d7b504170d071c77818105f7a07dc0297c6bea0c61a5404b071fed0c1fae`、text plan
  `sha256:0474ed893fbc043c3ace0197515f8d99e27fe3d28a4844fbdd9781bb9d30c7fa`、MTP plan
  `sha256:2965d8126e013cdd42f2c3764071b3dc34104fb8c43b3d94a9dfdb489ec4714b`を使った。
- canonical R9700 `gfx1201`とV620 `gfx1030`で、BF16+FP16 KVおよびFP8 W8A8+static FP8 KVのdraft width
  `1/2/3/6/7`、target block M=`2/3/4/7/8`を実行した。全caseでblock Argmax tokenとfinal hidden rowsが逐次M=1と
  bit-exact、partial-prefix replay一致、HIP-only、fallback false、cleanup 0だった。
- 最広M=8では全8 row x 248,320 vocabのraw BF16 logitsと、accepted prefixの全8 full-attention layer K/V semantic payloadを
  逐次M=1 oracleへbyte-exact照合した。R9700 digestはBF16 logits
  `sha256:00cc3fc0454a3052fcb9782bc2940dc6d536836acbee74204a19284e97450f0c` / KV
  `sha256:c35b8b3631b0ff2114f5826289ea8aacdaefbc0c2f486a3f1a57ea9590ffc747`、FP8 logits
  `sha256:4f6bb518a72c2b98c73e203156a9a786ceb3c07332333f2ae7f17de8864a2294` / KV
  `sha256:bd708ccb6d145dfb440b177cd39f4352bb34e5dedb778bb3dd23222b79582e94`だった。V620はBF16 digestがR9700と同じ、
  FP8 logits `sha256:88845231658336fd5be1c8e6f7960367abe7bb2b49178e691e9c70982c298569` / KV
  `sha256:d2de296eb5e1835ef7ff29b0c1581c55d06bb5c666706e23bb52c3d0f804f40d`だった。
- R9700 OpenAI production serverでnon-stream/SSE、連続request、client timeoutによるcancel、直後のrecovery、graceful shutdownを実行した。
  responseは`ok`、SSEはrole/content/final usage/`[DONE]`順、cancel auditは`outcome=cancelled`、次requestは`recovered`、shutdownは
  request/workspace/final current bytes 0、全request HIP-only/fallback falseだった。通常CLI `generate`も同じ内部選択でPASSした。

## 2026-08-16: paired性能と採用判断

- R9700 BF16、fixed Rust prompt、32 output、greedy、draft width 1を3 warmup + 10 measured、off/on counterbalancedで測った。
  output/finish/usageは全runでexact、fallback false、cleanup 0だった。最終binaryのspeedup中央値は`1.035546`、MAD
  `0.002847`、p10/p90 `1.024236/1.044816`、off-first/mtp-first中央値 `1.034016/1.036859`で、両実行順の中央値がとも改善した。
  1 run当たりproposal 16、accepted 15（0.9375/token）、target rows/output 0.96875、target dispatchは15,744から8,856、
  MTP dispatchは1,475だった。
- V620 draft width 1 screeningは中央値`0.999010`、MAD `0.002763`、p10/p90 `0.996800/1.001667`でnoise内だった。
  draft width 2も先行screeningで`0.9337x`だったため通常providerはtarget-onlyを維持する。
- よってPhase 18の最低一target採用条件はR9700で満たした。倍率はこのfixed model/prompt/length/tupleだけの値であり、他engineや
  別modelへ一般化しない。V620を「MTP未対応」とはせず、正確なM>1実装は保持したまま性能tableで非選択にする。

[対応する計画](../../../../plans/archive/2026/08/11-20/phase18-mtp-exact-sequential-speedup.md)
