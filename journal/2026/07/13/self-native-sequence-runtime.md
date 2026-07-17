# Qwen3.5 AQ4 self native sequence runtime

- `PackageSelfAttnSequenceWorkspace` をモデル全体で1組だけ確保し、Qwen9B geometry（M=128）を 39.5 MiB として checked admission した。
- self-attention の M>=2 を `[M,H]` の native pipeline（segmented RMSNorm、Q/K/V batch projection、Q/K norm+RoPE、typed paged KV/GQA chunk、O/residual、post RMS/MLP）へ接続した。
- writer 成功後だけ `written_len` を進め、reader/後段失敗は request state を poison。final row は layer output buffer へ D2D retain。
- block table entry range、gated Q projection、sigmoid-gate chunk reader、checked size を load/sequence 境界で検証した。
- 検証: self workspace unit 1 passed、model runtime 5 passed、backend registry 30 passed/1 ignored、`cargo check -p ullm-engine` passed。実機 HIP activation と forced M1 比較 hook は未実施/未実装。
