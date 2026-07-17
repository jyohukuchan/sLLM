# SQ plan update after FlashAttention prerequisite

## 前回の要点

- cached-prefix attentionは `cached_prefix_rdna4_fp8_auto` まで進み、SQ候補評価に使う暫定default executorとして扱える。
- `L=65536,M=128` はattention component単体で約1秒級になり、SQ計画を止めるほど遅くはない。

## 今回の変更点

- `docs/plans/fp8-sq-r9700-batch-throughput-prefill-plan-v0.1.md` を更新した。
- FlashAttention2-style実装を「次タスク」ではなく「完了済みの前提作業」として位置づけ直した。
- 次の主タスクを `sq-fp8-w8a16-r9700-v0` のpackage/runtime prototype、result schema固定、AQ4 baseline整理へ切り替えた。
- T0には `resolved_executor` 記録と `cached_prefix_rdna4_fp8_auto` の暫定default化を追加した。
- T2には artifact manifest、scale metadata、resident bytes、working-set bytes、passthrough tensor一覧の保存を追加した。

## 次の行動

1. T0: R9700 SQ evaluation state freeze noteを作る。
2. T1: batch throughput runnerのJSONL集約とVRAM/KV cache bytes記録を整える。
3. T2: `sq-fp8-w8a16-r9700-v0` のpayload writerとruntime load pathを作る。
