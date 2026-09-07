# Phase 78後の他モデル測定

2026-09-05のユーザー指示「他モデルでも計測して」に対応する追加観測。
Phase 78の完了判断は維持し、新しい性能目標やモデル固有最適化は追加しない。

## 測定範囲

- Qwen3.5-4B／9Bの既存locked MXFP8 E4M3 W8A8 GGUF、FP16 KV。
- canonical V620 gfx1030／R9700 gfx1201、単一要求、greedy、EOS無視。
- 入力はtoken 23066の17個／512個、出力17／32個。prefill chunkは入力長と同じ。
- 各条件1 warmup＋3 measured、旧版→現行版→旧版。中央値とばらつきを比較する。
- 現行CLIはcommit `40ab582b049cff7effadbca75fe951d6cef5bd96`からビルド。
  比較対象は保存済みPhase75 gfx1030／Phase74 gfx1201 CLIで、binary hashを固定する。

## 判定と制約

GPU実行成功、HIP-only、fallbackなし、生成token・停止理由・cleanupを確認して性能値を報告する。
失敗・未選択はGPU PASSへ含めない。旧版全体との比較であり、個別最適化の寄与率は断定しない。
MXFP8はFP8 outer-vectorとは異なる形式なのでID71の効果を測るものではない。
通常CLIのQwen embedded FP8 GGUF選択はgfx1030を受け付けず、今回その対応拡張は行わない。
raw結果、command、binary、hashはGit管理外の`.local-artifacts/phase78-cross-model/`へ保存する。

2026-09-05、全8条件・24 runを完了。生成token・停止理由・audit一致、HIP-only／fallbackなし／cleanup 0を確認した。
要求準備22～36%短縮を観測したが、推論全体の大幅改善は確認しなかった。

[測定履歴](../../../../../history/2026/09/1-10/phase78-cross-model-measurement.md) ·
[集約証拠](../../../../../history/2026/09/1-10/phase78-cross-model-evidence.json)
