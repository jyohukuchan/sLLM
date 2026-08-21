# Phase 37以降 MI300X最適化・llama.cpp機能差解消履歴

## 2026-08-21: 計画作成

- ユーザー指示により、Phase 36で残ったMI300X `gfx942`性能差と、main planに記録済みのllama.cpp比機能差へ
  Phase 37以降を割り当てた。
- Phase 37–38をMI300X性能laneとし、Session Dでdevice timeの`73.95%`を占めたGDN、`25.12%`を占めた
  Full Attention、続くfresh residualの順に扱う。
- Phase 39–48をservice基盤、token selection/grammar、state/cache、基本endpoint、Responses/Anthropic/tool protocol、
  template/CLI UX、adapter/model lifecycle、周辺tool、組込みtool/MCP、WebUIへ依存順に分けた。
- ユーザー方針どおりMI300X実機baseline/performanceはVM再確保までdeferredとし、Phase 37はhost prepだけを進行可能にした。
  Phase 39以降のhost実装はPhase 37/38のGPU完了を開始・merge gateにしない。
- Vulkan、一般INT4/INT8+scale、model/hardware/parallel追加は意図的除外を維持した。組込みtool/MCP実行は新しい
  security boundaryのため、Phase番号は割り当てるが実装開始にはtrust modelのユーザー承認を必要とする。
- focused reviewを反映し、fixed llama.cpp比較をpeer artifactが一致するBF16 weight＋FP16 KV行に限定した。FNUZ FP8は
  sLLM内BF16対照とし、対応peerなしに比率を作らない。resumable transport、`n` choice state、assistant prefill、FIM/infillは
  各一つの所有Phaseを定め、後続Phaseはwire/renderer/UX adapterだけを担当する。
- この時点では計画と文書同期だけで、production source、GPU、VM、external service、commit/pushを変更していない。

[対応する計画](../../../../plans/active/2026/08/21-31/phase37-plus-mi300x-and-llama-gap-roadmap.md)
