# Phase 23 cross-engine differential performance discovery history

## 2026-08-18: 詳細計画作成

- ユーザーの明示指示により、Phase 23をproduction最適化の実装ではなく、見落としている最適化余地を抽出する探索専用Phaseへ
  割り当てた。
- Phase 5のcross-engine baseline、Phase 21のsegment completion candidate棄却、Phase 22の局所matvec改善がwallへ
  転化しなかった結果、Phase Xでbuild coverageが根因だった事例を開始根拠として整理した。
- primaryをQwen3.5-4B BF16、canonical V620 `gfx1030` / R9700 `gfx1201`、primary peerを固定llama.cppとした。
  vLLMまたはSGLangはexact条件を満たすserving controlとして選び、比較不能な条件は速度比へ変換しない方針とした。
- cold load、warm direct/API、prefill、decode、concurrencyをbroad scanし、host span、HIP API/kernel/copy、GPU countersを
  observer effect別のlaneで細分化する計測モデルを定義した。
- overlapを二重計上しないcritical-path accounting、E0/E1/E2比較class、既存backlogの再分類、Amdahl上限と
  cost/risk/provenanceを含むopportunity ledgerを完了条件へ固定した。
- Phase 23は最大3件のPhase 24候補を提示するが、自動的に実装を開始しない。full-modelで5%以上の現実的改善余地は
  shortlist用のnonblocking AI提案として扱い、5%未満の候補の記録・提示・ユーザー選択を妨げない。
- この時点では計画、history、main planだけを更新した。計測tool、schema、runner、production source、GPU実行、
  外部engine build/source inspection、candidate実装はまだ開始していない。

## 2026-08-18: matched scan、critical-path分析、closeout

### Identityと比較可能性

- Qwen3.5-4B BF16 GGUF SHA-256 `c571c54eb8e2c9e935790d885e6d20f29c5fc82cd00ae28ddb5937a77c7fc675`、
  model revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`、ROCm 7.14、canonical V620 `gfx1030` / R9700
  `gfx1201`へ固定した。target別CLI/server release binaryのdigestを集計へ記録した。
- primary peerはllama.cpp commit `f5919bf458ef190468b5c329bb293f8a54a1e69c`のimmutable Phase 5 evidenceとした。
  256-token prefillはE1 system-equivalent、fresh decodeはtoken列・出力長が異なるためE2 diagnostic-onlyとした。
- vLLM/SGLangはmoduleとlocal container imageがなく、fixed environmentのmatched controlを作れなかった。固定revisionのscheduler
  sourceからfacts-only technical comparisonを作り、速度ratioを出さなかった。

### Fresh production wall

- warm short chatのTTFT/E2EはV620 186.77/473.55 ms、R9700 48.53/295.81 msだった。
- 256-token prefillはV620 2.270 s、R9700 317.58 msで、E2E比率は96.87%/80.81%だった。固定llama.cpp system resultとの
  gapは6.44x/6.60xだった。
- 28-token inputから128-tokenを完走するdecode caseはV620 32.43 tok/s、R9700 36.99 tok/sだった。post-TTFT wall shareは
  92.34%/98.12%である。条件が異なる固定peerとの勝敗ratioは作らなかった。
- HTTP non-stream/SSEのbackend外残差は約0.5〜0.6 msだった。同時2要求の完了時刻はV620 0.471/0.937 s、R9700
  0.325/0.651 sで、一つのwhole-generation workerによる直列化を確認した。
- fresh-process model-ready 5回の中央値はV620 10.53 s、R9700 11.60 sだった。OS page cacheは
  `uncontrolled-warm`と明記し、cold disk値へ読み替えない。

### Profilerと見落とし

- rocprofv3 observer laneではinstrumented wallをproduction値へ使わず、kernel/API/copy shareとdispatch shapeだけを採用した。
- generation adapterはprefill outputの最後のtoken IDだけを消費するが、graphは`M`行すべてのvocabulary projectionとargmaxを
  実行していた。256-token LM-head-shaped workはdevice timeのV620 13.48%、R9700 46.92%で、normal E2Eへ換算した
  Amdahl上限は13.06%/37.92%だった。これを新規最上位候補`P23-O1`とした。
- short/long profileと過去のexact comparisonからdecode projection familyを再確認した。Gemma 4 mixed low-bit R9700 controlも
  matvec 83.67%を示したため、単一shape tuningではなくfamily fusion/shared load/plan replayを`P23-O2`とした。
- service serializationを`P23-O3`、full-file hashとper-binding/chunk waitを含むcold loaderを`P23-O4`とした。
  terminal-row除去後のprefill provider/GDN再profileを`P23-O5`、HTTP/SSEとisolated event削減の低優先度化を`P23-O6`とした。

### Phase 24 shortlistとnegative findings

1. `P23-O1`: prefill last-row-only LM head/argmax。期待E2E改善はV620 8〜13%、R9700 20〜38%。
2. `P23-O2`: projection-family fusion/shared load/plan replay。Qwen decode期待値8〜20%、provider別検証が必要。
3. `P23-O3`: continuous batching。concurrency=2 aggregate throughput期待値30〜80%、single request改善は目的外。

- HTTP/SSE framing、短context full attention最優先化、requestごとのgraph instantiate、Phase 21型のisolated event削減は
  fresh evidenceからPhase 24最上位へ置かなかった。
- Phase 23はproduction source/default/API/model formatを変更せず、探索と候補順位付けだけで完了した。
- schema/summary validation、runner contract、focused unit test、format/diff、Markdown link、license/provenance、
  related Rust host checks、integration reviewを実施し、correctness/security blockerは残さなかった。
- provenance validatorが固定`imported_sha256`を現在の保守後sourceへ誤って比較していた既存不整合を検出し、policyどおり
  記録済みimport commitのblobへ照合するよう修正した。現在sourceのprovenance header検証は維持した。

[対応する計画](../../../../plans/archive/2026/08/11-20/phase23-cross-engine-differential-performance-discovery.md)
[集計JSON](../../../../../ci/matrix/phase23-performance-discovery-summary-v1.json)
[engine差分note](../../../../references/phase23-inference-engine-performance-differential.md)
