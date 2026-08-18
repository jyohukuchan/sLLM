# Phase 25 batch-compatible projection-family optimization history

## 2026-08-18: Phase 23/24結果を受けた詳細計画

- ユーザーの明示指示により、Phase 23 shortlistの`P23-O2`をPhase 25へ割り当てた。
- Phase 22では単一`M=1` matvecのoperator改善がfull-model wallへ転化せず、Phase 24ではR9700の`M>1` hipBLAS GEMMから
  `M=1` decode reductionへのprovider遷移がterminal-row削減を相殺した。このため単一shape tuningを再提案しない。
- primaryをQwen3.5-4B dense BF16、canonical `gfx1030`/`gfx1201`、通常decode projection familyとした。
- Q/K/V、gate/up/down、terminal projection、prepared planをfresh critical-path shareから順位付けし、一度に一familyだけ実装する。
- Phase 26でdecode row数が`M=B`へ変わることを前提に、host/GPU contractは`M=1,2,3,4,7,8,16,17`を含める。
  Phase 25自身はscheduler、per-sequence state、production request batchingを実装しない。
- 共通semantic/graph pathを優先し、target差は実測で必要な場合だけkernel registryのprovider selectionへ閉じ込める。
- 提案採用基準は全固定target/patternでstableなfull-model悪化なし、任意target/patternでE2EまたはTPOT 5%以上改善とした。
  operatorだけの改善、profiler wall、synthetic `B>1`はproduction採用値にしない。
- 本更新は計画のみである。source、schema/runner、GPU evidence、baseline/candidate、production defaultは変更していない。

## 2026-08-18: Phase 24後profileとnegative closeout

- ユーザーのPhase 25/26完了指示により、Phase 24採用後のcurrent sourceをexact `gfx1030`/`gfx1201`向けに再buildし、
  Qwen3.5-4B BF16 short decodeをrocprofv3 runtime traceでfresh profileした。両runは同一token列、10 generated token、
  9 decode step、HIP-only、fallbackなしで完了した。
- projection device shareはV620 86.48%、R9700 79.23%だった。ただし支配時間の大半はprojectionごとに異なる必須weight readであり、
  family fusionが除去できる量ではない。
- 最大の連続shared-input候補であるgate/upは一layerあたり94,371,840 bytesのweightに対し、共有できるactivation readが
  `M=1`で5,120 bytes、0.00543%に過ぎない。32 layerで一launchずつ完全に消す楽観上限も、observer effectを含む
  rocprofのlaunch平均を使ってなおTPOTのV620 0.94%、R9700 2.60%だった。
- linear-attention projectionは既にpackedで、残るfull-attention Q/K/Vのlaunch削減はgate/upより小さい。wide-vocabulary `M=1`は
  現行decode reduction provider上にあり、正しいcandidateがweight workを除去できない。prepared descriptorも既にcacheされ、
  reusable prepare/replay spanは5%へ届かなかった。
- 従って「family share」ではなく`critical-path share × credible removable fraction`で評価すると、固定した5% full-model採用条件へ
  到達可能なwork unitがない。計画のA0停止条件によりproduction candidateを実装せず、target分岐も追加せず、Phase 25を
  negative discoveryとして完了した。
- raw trace/binaryは追跡せず、digest-bound aggregate、schema、host validationだけを保持する。`M=B`でarithmetic intensityとproviderが
  変わるPhase 26は独立に開始でき、今回の`M=1`負結果をrequest batchingの否定には使わない。

[対応する計画](../../../../plans/archive/2026/08/11-20/phase25-batch-compatible-projection-family-optimization.md)
[bounded summary](../../../../../ci/matrix/phase25-projection-family-summary-v1.json)
