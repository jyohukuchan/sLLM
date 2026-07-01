# AQ codebook alternating optimization

## 作業

- `tools/run-aq-codebook-opt-experiment.py` を追加した。
- `codebook` FP16、`local-scale` 離散値、`global-scale` FP16/FP32 を前提に、sampled block 上で交互最適化するCPU実験を実装した。
- `gpt-5.3-codex-spark` worker に初期実装を任せ、僕が `codebook-index` 更新、最小二乗 codebook 更新、final assignment、thread設定をレビュー修正した。

## 測定

- Qwen3.5-9B BF16 safetensorsを使用した。
- E4M3 g16/g8, 262k elements, 3 iterations, 4 families x 2 tensors:
  - mean alternating/baseline relative MSE: `0.986642`
  - elapsed: `12.31 s`
  - max RSS: `520916 KiB`
- E4M3 g16/g8, 1M elements, 3 iterations, 4 families x 1 tensor:
  - mean alternating/baseline relative MSE: `0.987425`
  - elapsed: `24.01 s`
  - max RSS: `580316 KiB`
- E4M3 g16/g8, 262k elements, 8 iterations, 4 families x 2 tensors:
  - mean alternating/baseline relative MSE: `0.976375`
  - elapsed: `22.23 s`
  - max RSS: `518124 KiB`
- FP16 global-scaleとFP32 global-scaleの差は平均 relative MSE delta `-9.70e-08` 程度で、今回の条件では実質的に無視できる。

## 結論

- 現実の保存制約を入れた codebook 交互最適化はCPU sample実験として十分現実的。
- 8 iterations では代表familyでBF16-errorが約2.4%下がった。
- E8M0は交互最適化による改善率は大きいが、絶対BF16-errorはE4M3/UE5M3より悪い。
- 同時最適化の完全brute forceは現実的ではない。固定codebook/global-scaleでの全local-scale探索だけでもE4M3でbaselineの約13.22倍、E8M0/UE5M3で約28.33倍のoperation countになる。

## 参照

- 詳細: `docs/research/aq-codebook-optimization-results-2026-07-02.md`
- 結果JSON: `benchmarks/results/2026-07-02/aq/`
