# Phase 27 exact decode projection weight-stream/provider optimization history

## 2026-08-18: Phase 23/25の未確認範囲を受けた詳細計画

- ユーザーの明示指示により、Phase 27を通常decode projectionの必須weight stream・演算provider最適化へ割り当てた。
- Phase 23のfresh sLLM long decodeはV620 32.43 tok/s、R9700 36.99 tok/sだったが、fresh llama.cppはtoken列と出力長が
  異なりE2だった。historical exact laneのthroughput ratio 0.724/0.712がcurrent sourceでも残るかを最初に再検証する。
- Phase 25で否定した範囲をprojection-family shared activation、launch除去、plan replayに限定した。projection時間が
  V620 86.48%、R9700 79.23%でも、必須weight streamの実効帯域、provider、layout、wave/vector mappingがpeer相当とは限らない。
- primaryをQwen3.5-4B dense BF16、canonical `gfx1030`/`gfx1201`、warm single-request、batch 1、MTP offとした。
  同一prompt/continuation token IDsをstepごとに与えるteacher-forced direct replayをE0 primaryとし、tokenizer/sampler差を除外する。
- Q/K/V/O、linear/GDN、gate/up/down、LM headをrole/shape/providerへ分解し、mandatory bytes、effective weight-stream rate、
  DRAM/L2/VMEM/occupancy/stall、host gapを別laneで計測する。単純なprojection shareではなくgap contributionとAmdahl上限から
  一candidateだけを選ぶ。
- Phase 22 candidateの無計測復元、P23-O2 fusionの再実施、prefill、batching、quantization、DeepSeek V4、TurboQuant、
  runtime同期、cold loaderを非対象とした。projection外が主因ならPhase 27を拡張せず別候補へ戻す。
- 採用規則は全固定target/patternで悪化なし、任意target/patternでfull-model decode TPOTまたはpost-TTFT E2E 5%以上改善とした。
  cross-engine gapはfixed-token E0、production採否は通常greedyのsLLM baseline/candidateで分ける。共通semantic pathを優先し、
  target差は共通provider registryのselectionへ閉じ込める。
- fresh gapが5%未満、比較がE0にならない、最大候補の上限が5%未満、またはcandidateがfull-modelへ転化しない場合は、
  production変更を残さずnegative completionできる。
- 本更新は計画のみである。source、runner、schema、GPU evidence、llama.cpp build、baseline/candidate、production defaultは変更していない。

## 2026-08-18: fresh比較、critical-path attribution、negative closeout

- Qwen serviceを停止してcanonical V620 `GPU-76a08c022586fed6` / R9700 `GPU-a8e9ddefa2d60f55`を解放し、ROCm 7.14、
  LLVM 23、exact `gfx1030`/`gfx1201`でcurrent sLLMをfresh buildした。stub/public-runtime無効buildとコンパイル中に重なった試走は
  FAILまたは破棄し、採用証拠に含めなかった。
- llama.cpp `f5919bf458ef190468b5c329bb293f8a54a1e69c`をdetached temporary worktreeから両targetへfresh buildした。
  HIP libraryのoffload imageは各exact targetだけで、Qwen3.5-4B BF16 artifact revisionはsLLMと同じ`851bf6e...`だった。
  GGUF digestはsLLM `c571c54e...`、llama.cpp `636158bd...`で異なるため、logical system-equivalent E1へ降格した。
- 28-token prompt / 128-token greedyでsLLMはV620 32.3838 tok/s（MAD 0.2372）、R9700 36.9990 tok/s
  （MAD 0.1554）だった。llama.cpp decode-only 10 repetitionは48.9403/53.9641 tok/sで、throughput ratioは
  0.6617/0.6856だった。gapはhistorical exact laneと同方向だが、token列とtiming boundaryが一致しないためcurrent engineの
  exact勝敗とは表記しない。両sLLM runはHIP-only、fallbackなし、cleanup terminal-zeroをPASSした。
- 別のrocprofv3 runtime traceで同じ249 projection/tokenを集約した。mandatory BF16 weightは8,409,579,520 bytes/token、
  V620のsLLM/llama.cpp projectionは17.711/18.994 ms、effective stream proxyは474.82/442.74 GB/sだった。
  R9700は17.852/15.864 ms、471.06/530.11 GB/sだった。V620ではsLLM projectionがpeerより6.76%速く、R9700だけ
  12.53%遅かった。
- projectionを除くcoarse residualはV620 5.379/1.414 ms、R9700 5.257/1.485 msだった。後続監査で、sLLM側はprefillの
  projectionだけを引き、prefill中のGDN/attention/normとR9700のMTP内部workを残していたことを確認した。llama.cppともstep境界が
  一致しないため、decode projection外device timeの3.80倍/3.54倍という解釈を撤回し、Phase 28の再計測仮説へ限定した。
- R9700 projectionをpeer相当まで完全に短縮する楽観上限はproduction TPOTの約7.36%だったが、V620では同じ置換の余地がない。
  Phase 22 wave32x8はR9700 operatorを約13%悪化させ、V620 gate-only full-modelも0.52%悪化した。Phase 25でboundedに棄却した
  shared activation/launch/fusionをPhase 27で再開しないため、共通providerで全target非悪化かつ任意pattern 5%改善へ届く
  credible candidateを固定できなかった。
- production source、default provider、target分岐を変更せず、Phase 27をnegative discoveryとして完了した。raw model、binary、
  trace DB、generated textは追跡せず、schema-validな[bounded summary](../../../../../ci/matrix/phase27-weight-stream-summary-v1.json)へ
  identity、aggregate、comparison limit、再検討条件だけを固定した。

[対応する計画](../../../../plans/archive/2026/08/11-20/phase27-exact-decode-projection-weight-stream-provider-optimization.md)
