# 2026-07-27 朝の引き継ぎ — AQ4_0 overnight consolidation

## 結論

Qwen3.5-9B `AQ4_0` の本番 worker を、依頼CCの終了後に確定した
`840a1c7a2fecef6063433b7ffc96b9298840154f` から隔離 worktree で1本だけ
build し、R9700 (`gfx1201`) で測定・固定10プロンプト検証の後に昇格した。
CCは本番 manifest を変更せずに終了したため、昇格の rollback 起点は従来の
`3507102…` manifest である。

最終 active manifest SHA-256 は
`a654d92fe8142fcc0904fe187c96b84c95e0dd18acac61ef25d0cfa6429a08cd`。
served worker は
`/opt/ullm/aq4-overnight-consolidation-v0.1/releases/aq4-consolidated-840a1c7a-5a274733/ullm-aq4-worker`
（`root:root`, `0555`、SHA-256
`5a274733710d9b80a24d34a31ec6a99ac0b2d1e8fcce45904e906926a0e2e903`）である。

`worker.execution.paged_decode_attention` は
`aq4_gqa_grouped_split` / `split_tile: 128` を維持した。K/V cache は
`ULLM_KV_CACHE_DTYPE`、`ULLM_KV_CACHE_TYPE_K`、`ULLM_KV_CACHE_TYPE_V` の
いずれも未設定で、コード既定値の F32 を使用している。

## 統合した main の改善

以下が build base の祖先であることを確認した。

- `cd7c1768`（CA）: AQ4 matvec-add 最適化。今回の grouped decode 実測は約
  78 tok/s で、従来本番の 74.591159 tok/s には留まらなかった。
- `f6b58e6c` / `2031e968`（BY）: prefill chunk 幅 M の可変化。
- `b46cc8ac`（BR）: prefill Flash2 の GQA staging。
- `1c660223`（CC）: prompt 長に応じた adaptive prefill width。
- `d8389e59`（BX）: native FP16 / FP8 (S1E4M3) K/V kernels。

build base 後に `main` は `95548add4e5c208ee8bf017e5e0ecdea6d95779a` へ進んだが、
差分は MoE の計測記録と journal のみで runtime source は変更していない。従って
新しい worker を再buildする理由はなかった。

## 実測と生成品質

R9700 を service 停止中に専有し、プロファイラ range ではなく profile binary の
full-model wall-clock JSON を throughput として記録した。

- Decode: C=1339、warmup 6、32 measured step × 2。77.908833 と
  77.763992 tok/s、平均 **77.836412 tok/s**（weighted: 77.836345 tok/s）。
  期待した約78 tok/s に到達しており、CA の改善が worker に入っていることと整合する。
- Prefill: p2048 / M128、**975.421658 tok/s**（2.099604805 s）。970–1,020
  tok/s の参照帯を下回らなかった。

`tools/promote-served-model.py` による固定10プロンプト比較は、baseline / candidate
とも HTTP 200 で完了した。candidate は **10/10 exact match**、blocking finding 0、
`passed: true`。日本語、英語、コード、要約、多ターン、翻訳、reasoning のすべてに
実応答があり、tool の actual-response probe も通過した。これは policy に従う生成品質の
確認であり、速度の数値しきい値を昇格 gate には使用していない。

## 本番に有効化していないもの

`d8389e59` の native FP16 / FP8 (S1E4M3) K/V path は source と worker には含まれるが、
本番では**有効化していない**。served-model 側の fail-closed K/V dtype selector が未整備で、
prefill 回帰も記録されているためである。F32 を維持した。

それ以外の上記 runtime 改善を意図的に外したものはない。旧 CA candidate artifact は
個別に選ばず、同じ最適化を含むこの統合 build に置き換えた。

## サービスと競合回避

- GPU 計測 window: `ullm-openai.service` を1回 stop し、1回 start して復旧。
- 昇格 tool: `restart` 1回、start-limit recovery は不要。
- 合計の service 起動/再起動操作は2回。現在の `NRestarts=0`、
  `ullm-openai.service` は `active`、`llama-qwen35-udq4.service` は
  `inactive` / `disabled`。
- active manifest は昇格直前に旧 SHA `3507102…` であることを確認し、tool は
  baseline 後にも byte drift を検査してから atomic swap した。昇格後の active SHA は
  上記 `a654d92…` で再確認済み。

## 証跡

- [統合 evidence README](../../benchmarks/results/2026-07-27/overnight-consolidation/README.md)
- [R9700 測定要約](../../benchmarks/results/2026-07-27/overnight-consolidation/gpu-window-20260727T075720+0900/measurement-summary.json)
- [昇格 outcome](../../benchmarks/results/2026-07-27/overnight-consolidation/promotion-20260727T080228+0900/outcome.json)
- [10 prompt 比較](../../benchmarks/results/2026-07-27/overnight-consolidation/promotion-20260727T080228+0900/comparison.md)

## 次に必要な作業（優先順）

1. 本番 manifest の外部上書きを防ぐ運用を決める。新しい active SHA と immutable
   worker path を deployment ownership の正本として共有し、別セッションは昇格前に
   SHA を確認する。
2. K/V FP16 / FP8 (S1E4M3) を試す場合は、served-model manifest に fail-closed selector
   を実装し、F32 baseline と prefill・生成を比較してから別 candidate として昇格する。
3. adaptive prefill width は本番 prompt 長分布で追加測定する。今回の p2048/M128 は回帰
   なしの spot check であり、全長域の性能優位は未確認。
4. runtime source を変更する後続 commit が入った場合だけ、同じ隔離 build → 短い
   R9700 window → lightweight promotion の順で新しい immutable release を作る。
