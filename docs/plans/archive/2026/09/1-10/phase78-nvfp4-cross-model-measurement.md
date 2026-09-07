# Phase 78後の他モデルNVFP4測定

2026-09-05のユーザー指示「NVFP4重みの場合は？」に対応する追加観測。
既存のGemma 4 12B NVFP4／FP8混合GGUFを使い、前回のMXFP8測定を補う。

- locked `phase20-final-gemma4-nvfp4.gguf`、FP16 KV、単一要求、greedy、EOS無視。
- token 23066の17個／512個を入力し、17個／32個出力。Gemma CLIはchunk指定を受け付けないため指定しない。
- V620 gfx1030／R9700 gfx1201。前回と同じ旧版・現行CLIをhashで固定して再利用。
- 旧版→現行版→旧版、各1 warmup＋3 measured。全4条件の中央値・ばらつき、token／stop／audit、HIP-only、cleanupを確認する。
- 全体比較をNVFP4演算だけの寄与へ読み替えない。modelが混合重みであること、共通selectorとshape限定opt-inを区別する。
- rawは`.local-artifacts/phase78-nvfp4-cross-model/`へ保存。新しい最適化・対応拡張・必達速度は追加しない。

2026-09-05完了。V620 512/32は旧版runが640秒となったため旧版／現行各1+3へ再計画し、旧版の再反復を省略した。
進行中のGPU計算は正常終了まで待った。主4条件とNVFP4 rollback／汎用DP4Aの追加切り分け、計17 runを完了。
共通prefillの速度効果は観測したが、token差があるため品質維持の汎用採用とは扱わない。

[測定履歴](../../../../../history/2026/09/1-10/phase78-nvfp4-cross-model-measurement.md) ·
[集約証拠](../../../../../history/2026/09/1-10/phase78-nvfp4-cross-model-evidence.json)
