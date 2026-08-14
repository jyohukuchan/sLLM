# Phase 12待機中のローカル先行実行キュー履歴

## 2026-08-15: Phase 12Rを先頭へ追加

- ユーザー指示によるCI修正をPhase 12Rへ割り当て、local forward queueの先頭へ追加した。
- Phase 12R、Phase 13、Phase 14、cross-model RDNA性能bridge、Phase 15、Phase 16以降の順とし、既存Phase番号は
  繰り下げなかった。
- Phase 12RはMI300X実機PASSやPhase 12完了を主張せず、GitHub host portabilityとlocal GPU laneの修復に限定した。
- ユーザー指示により、各Phaseのcloseout後にPhase単位でcommit・pushし、次Phaseの変更を混ぜない運用を追加した。

## 2026-08-15: Phase 12R完了、Q1へ移行

- tracked-only host portability、H3 link closure、manual self-hosted trigger、registry-driven local entrypointを完了した。
- Phase 12とMI300X evidenceは`ready`のまま維持し、次のlocal work unitをQ1 Phase 13へ進めた。

## 2026-08-15: queue作成

- ユーザーがMI300X cloudを管理できない十数時間以上に、local-only workを`/goal`で継続できるようにした。
- Phase 12を`ready`のまま保持し、Phase 12R、Phase 13、Phase 14 Gemma 4 Dense、cross-model RDNA性能bridge、Phase 15
  Weight NVFP4の順に進むqueueを固定した。
- Gemma 4 Denseをgoal終端にせず、Phase 16 KV量子化、Phase 17 MTP/vision、Phase 18 MoEの詳細計画・実装へ
  続く枯渇防止tailを追加した。
- Hot Aisle VMを起動しない境界と、帰宅後にlatest mainからPhase 12 candidateを再buildする規則を明記した。

[対応する計画](../../../../plans/active/2026/08/11-20/phase12-wait-local-forward-queue.md)
