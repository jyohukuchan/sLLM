# Phase 5: エンジン性能baseline計画

## 目的

OpenAI-compatible APIを追加する前に、model-resident lifecycleを再利用するsLLM direct engineの
性能を再現可能な条件で測定する。Qwen3.5-4B BF16を主baselineとし、TTFT、prefill、TPOT、
decode token/s、end-to-end latency、peak VRAMをcanonical AMD GPUで記録する。同じmodel revisionと
数値条件のllama.cppを比較peerにし、最適化前の差とperformance cliff候補を可視化する。

本計画はbaseline取得であり、任意の高速化倍率やP1回帰thresholdを先に設定しない。測定値を
correctness、GPU対応、または最適化済みの証明へ読み替えない。

## 前提と依存関係

- [Phase 4 Qwen3.5-2B・9B互換性確認計画](../../../../archive/2026/08/11-20/qwen35-2b-9b-compatibility.md)は
  integration完了済みであり、本計画を開始可能とする。
- 主modelは現行Qwen3.5-4B lock fingerprint
  `sha256:f143d7b504170d071c77818105f7a07dc0297c6bea0c61a5404b071fed0c1fae`とする。
- 比較peerはlocal固定llama.cpp commit
  `f5919bf458ef190468b5c329bb293f8a54a1e69c`とする。
- canonical GPUはV620 `gfx1030`とR9700 `gfx1201`。exact tuple、ROCm、kernel、clock/power、
  device healthを各runへ結合する。
- API実装後のservice/API overhead測定は本計画のschemaを再利用する別follow-upとし、
  このbaselineを待たせない。

## 測定対象と非対象

### 対象

- model loadからmodel-readyまでのone-time時間とresident VRAM。
- modelを再loadしないfresh request-local sessionのprefill、first token、decode、終了。
- pretokenized direct-engine laneとrender/tokenizeを含むCLI end-to-end lane。
- Qwen3.5-4Bのcanonical dual-GPU baseline。
- 2B/9Bの代表caseによるsize scaling確認。
- 同一4B source revisionから作ったllama.cpp BF16 artifactとの同条件比較。

### 非対象

- HTTP、SSE、TLS、JSON serializationなどAPI service overhead。
- request batching、chunked prefill、multi-stream、multi-GPU。
- CPU実行からのGPU性能推定、CPU fallback、GPU kernel emulation。
- benchmark結果を見た後のthreshold設定や、単一runの最良値による比較。
- profiler raw trace、binary、model、変換済みGGUFのGit管理。

## 指標定義

| 指標 | 定義 |
| --- | --- |
| model load | lock/cache検証開始から全required weight uploadとmodel-ready publicationまで |
| resident VRAM | model-ready後、request未作成時のdevice-local allocation high-water mark |
| TTFT | direct laneのrequest開始から最初のgenerated token publicationまで |
| prefill time | prefill submit直前からprefill completionとfirst-token publicationまで |
| prefill throughput | input token数 / prefill time |
| TPOT | 2 token目以降の隣接token publication間隔。各requestのmedianと分布を記録 |
| decode token/s | `(generated_tokens - 1) / (last token - first token)`。1 token caseでは未定義 |
| end-to-end | request開始からstop reason、visible output、cleanup完了まで |
| peak VRAM | runtime allocator high-water markを正とし、AMD SMI sampling値を補助記録 |

direct laneではmodel loadとtokenizeをTTFTから除外し、CLI laneではrender/tokenizeを含む。
GPU同期は指標境界でだけ行い、各kernelへ計測用同期を挿入して通常実行を変えない。

## case set

Qwen3.5-4Bで次を各targetへ実行する。promptはlocked tokenizerでexact token IDsへ固定し、同じ
token列を全engineへ与える。

| case | input tokens | requested output tokens | 目的 |
| --- | ---: | ---: | --- |
| minimum | 1 | 1 | 起動・first-token下限 |
| short-odd | 17 | 17 | 非整列の短い対話 |
| boundary-255 | 255 | 64 | 256境界直前 |
| boundary-256 | 256 | 64 | 256境界 |
| boundary-257 | 257 | 64 | 256境界直後 |
| prefill-long | 1024 | 128 | prefill寄り |
| decode-long | 32 | 256 | decode寄り |

各processでmodelを一度だけloadし、3 warmup requestの後に10 measured requestを実行する。
run順は記録し、model/request stateは毎回同じ初期状態へ戻す。tracked summaryはmedian、p10、p90、
MAD、min/max、sample countを持ち、raw sampleは`.local-artifacts/`へ置いてdigestだけを追跡する。
`prefill-long`はV620で5400秒の初期boundに到達した実測を根拠に、両GPUとも10800秒をbounded
timeoutとする。他caseのboundは変更せず、timeoutしたsampleを成功値へ変換しない。

## 作業単位

### P1: timingとmemory instrumentation（完了）

1. CLI専用timingを、model load、model-ready、request start、prefill submit/complete、first token、
   各decode token、stop、cleanupのmonotonic eventへ分解する。
2. model-resident ownerをprocess内で再利用し、request-local stateだけを反復作成・破棄する
   benchmark entrypointを追加する。
3. runtime allocationにmodel/request/workspace categoryとhigh-water markを追加し、report schemaへ
   exact bytesを出す。
4. timing/report schema、matrix、bounded runner、aggregate、raw summary digestを追加する。
5. fake clock、synthetic event、overflow、missing event、negative duration、sample 0件をhost testで拒否する。

受入条件:

- instrumentation off/onでtoken列とdispatch auditが同一である。
- model loadがmeasured requestごとに再実行されていない。
- TTFT、TPOT、E2Eのevent順と算術をhost testが検証する。
- report欠落、stale identity、別model/target、sample不足をaggregateが拒否する。

### P2: sLLM canonical baseline（完了）

1. exact build/model/GPU tupleを固定し、実行前後health、temperature、clock、power profile、
   resident process、ROCm loader pathを記録する。
2. V620 `gfx1030`とR9700 `gfx1201`で全caseを直列実行する。
3. correctness用golden caseを先にPASSさせ、性能runではtoken列、stop reason、fallback、dispatch、
   cleanupを監査する。
4. 2Bはshort-odd/boundary-257、9Bはminimum/short-oddを追加し、model size scalingだけを記録する。

受入条件:

- 全sampleがHIP backend、exact target、fallbackなし、timeoutなし、health/cleanup PASSである。
- model load、resident/request/peak VRAM、TTFT、TPOT、token/s、E2Eが全caseに揃う。
- 測定中の公開されたactive violation、slowdown温度到達、profile/limit/performance-level drift、他process干渉は成功値へ混ぜず、runを無効として理由を残す。socket powerはprofile/limitを変更せず全値を監査用に保持するが、R9700で公開cap/maxを超える瞬時telemetryを再現したため値単独をhard gateにしない。reasonを公開しないRDNA4のlegacy aggregate `throttle_status`も単独でthrottlingの根拠にしない。

### P3: llama.cpp同条件比較（完了）

1. 4Bの同じHugging Face revisionからllama.cpp固定commitの公式変換toolでBF16 GGUFを生成する。
   source lock fingerprint、変換tool commit/path/hash、引数、環境、output SHA-256をmanifestへ記録する。
2. conversion時間とartifactは推論指標から除外し、checkout外へ保存する。
3. `llama-bench`でprefill/decodeの比較可能な指標を取得し、generation end-to-endは固定promptと
   greedy/stop条件を揃えた専用wrapperで測る。
4. batch 1、context、input/output token数、BF16、GPU target/offload、warmup/sample数をsLLMと一致させる。
5. runner固有のmetric定義差を表に残し、直接比較できない値を比率へ変換しない。

受入条件:

- 比較表がmodel source revision、両artifact hash、両engine commit、GPU tuple、case、metric定義を持つ。
- 両engineのmedianと分布を並べ、単一最良runや異なるtoken数を比較しない。
- sLLMが遅いcaseも省略せず、原因仮説は測定事実と分離する。

### P4: baseline確定と最適化backlog（完了）

1. model/GPU/case別summaryとInferenceXと比較可能なgraph用tableを作る。
2. cliff候補は255/256/257、prefill/decode比、VRAM step、GPU差から抽出する。
3. baseline時点ではP1 hard thresholdを設定せず、履歴run数、分散、再現性が揃った後の
   nonblocking follow-upとしてthreshold案を記録する。
4. API実装後に同じdirect laneを再実行せず、service laneを追加してAPI overheadを差分測定できる
   schemaを確定する。

## 検証lane

- Draft: fake timing/memory、単一case、単一GPUのfocused測定。
- Integration: affected host suite、全canonical case、dual-GPU sLLM baseline、1回のintegration review。
- Release/push: clean identity、llama.cpp comparison manifest、全summary/digest、累積review。
- Docs-only: metric定義、link、summary整合のみ。性能runを取り直さない。

## Rollbackと停止条件

- instrumentationは通常generation pathから無効化できるadditiveな構造にし、token semanticsを変えたら
  baseline candidateを受け入れない。
- benchmark harness、llama conversion/comparison、summaryを別々にrollback可能にする。
- GPU health、throttling、他process干渉、cleanup失敗を成功sampleに変換しない。
- 同一failureが2回、1時間以上機能進捗なし、見積り1.5倍超過、またはmetric定義変更時は
  追加sampleを止めてreplanする。

## 完了結果

- sLLM directは22/22 row・220/220 measured sample、render/tokenizeは2/2 row、llama.cpp dedicated
  wrapperは14/14 row・140/140 measured requestがPASSした。
- direct aggregate SHA-256は
  `2fdf7b2fec8a50a0322b28d6be04effd40d68c203bceda7ed5438249fa490b7f`、render aggregateは
  `adb2cd63bf57af78f4d84a0bce64e6121eef1a041085512202df847cf672c553`、cross-engine comparisonは
  `53845c6501e78357b9b75ddcd8f960b2499cdbd64c94a2799afb95043799dccf`である。
- 4B sLLM TTFTはexact-token llama.cpp wrapperよりV620で49.4〜278.5倍、R9700で31.4〜742.1倍長い。
  最大差がminimumでなく255〜1024 token prefillへ拡大するため、最初の最適化対象は全model/GPU共通の
  prefill GEMM、operator dispatch、同期削減とする。
- 255/256/257に大きなcliffはなかった。境界3点は境界へ影響する変更だけで実行し、通常iterationでは
  4B short-oddと32/32短縮caseをwarmup 1 + measured 3で使う。minimum、2B/9B、render、llama比較、
  canonical long 10 sampleはintegration/release/nightlyまたは意味変更時へ限定する。
- P1性能回帰thresholdは単一baselineから設定しない。O2/O3で履歴runと分散が揃った後のnonblocking
  follow-upとし、Phase 6をblockしない。
- 詳細なmodel/GPU/case表、size scaling、比較倍率、失敗とhealth契約修正、O0〜O3 laneは
  [対応する履歴](../../../../../history/2026/08/11-20/engine-performance-baseline.md)を正とする。

## 完了後

baseline identity、metric定義、summary、llama.cpp比較条件、最適化backlogをmain-planとhistoryへ記録し、
本計画をarchiveする。その後、[Phase 6 OpenAI-compatible Chat Completions v1実装計画](openai-chat-completions-v1.md)へ進む。

[対応する履歴](../../../../../history/2026/08/11-20/engine-performance-baseline.md)
