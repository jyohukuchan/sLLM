# SQ8_1 W8A8 full-model 品質ゲート

## 前回の要点

- SQ8_1 K=32 signed symmetric int8、zero-point なし、upward-rounded FP16 scale の
  実装、artifact、W8A16 default / W8A8 explicit-only ABI は完成していた。
- activation-only CPU measurement は K=32 の activation relative L2 0.00994415、
  sampled W8A8 linear-output relative L2 0.00775109、16-token logit smoke の top-1
  16/16 一致を示したが、weight と activation を同時量子化する full-model gate は未確認だった。
- V620 の prequant prototype は W8A8 の性能余地を示したが、品質 admission の代替ではない。
  full-model quality を通過するまで prequant API へ投資しない方針だった。

## 今回の変更点

- Qwen3.5-9B を CPU FP32 reference として、248 transformer projection の W8A16 /
  W8A8 fake-quant full-model gate を追加した。W8A8 は weight と各 selected Linear
  input の双方を SQ8_1 K=32 で int8 化し、FP32 へ再構成して同じ FP32 `F.linear`
  boundary を通す。249th `lm_head` を加えた all-Linear stress は primary scope と
  分離した。
- frozen D_stats corpus の 20 deterministic records（5 domains x4）を測った。初回の
  256-token cap は 3,568 scored positions で、凍結済み 4,000 coverage 条件に届かず
  non-qualifying とした。raw artifact を `attempt-1-coverage-incomplete/` に保存し、
  threshold を変更せず同一 IDs の cap を 384 に上げた v0.2 qualifying run を
  4,243 positions で完走した。
- control は logits / final hidden の relative L2 と max abs が全て 0、weight / activation
  の post-storage clipping も 0 だった。GPU / HIP / CUDA / service は使っておらず、温度記録は
  N/A (CPU-only) である。
- W8A16 は fallback L2 gates を通過した（aggregate logits relative L2 0.016971283、
  worst prompt 0.056227554）。W8A8 は aggregate logits relative L2 0.023506802、
  mean KL 0.000665853、top-10 overlap 98.378506%、W8A16 比 1.385093 を通過したが、
  logits max abs 7.889154、final hidden max abs 13.696337、top-1 4,189/4,243
  (98.727316%、Wilson 98.343243%) で frozen gate を落とした。
- W8A8 の 54 top-1 mismatch 中 38 は FP32 top-2 への near-margin swap だったが、16 は
  predeclared rule を満たさず、AQ4 と同じ限定許容を適用しても採用できない。全 reference
  top-1 は W8A8 top-10 に残った。
- layer relative L2 は W8A8 layer 0 の 0.00796172 から layer 30 の 0.03357130、
  final norm 0.03126359 へ増えた。late-layer max error も観測したが、その internal
  mechanism は未確認である。
- `[4,8)` activation outlier blocks は 14.326868%、`[8,inf)` は 0 だった。
  source FP32 block を bypass する diagnostic は 14.331775% の blocks で activation
  relative L2 を 0.009489628 から 0.004431349 に下げ、W8A8-to-W8A16 logits-L2 gap を
  100% 除去した。これは side route の有望な根拠だが、同 diagnostic も max-abs /
  greedy 条件を満たさず、単独では full gate を救えない。
- 結論は **W8A8 No-Go**。runtime/artifact/release/campaign/authorization/active manifest
  を変更せず、W8A16 を required fallback として維持する。完全な evidence は
  `benchmarks/results/2026-07-26/sq8_1-w8a8-full-model-gate/` に保存した。

## 次の行動

1. W8A8 prequant API と production admission は開始せず、W8A16 default / W8A8 explicit-only を維持する。
2. K=32 `max(abs)/RMS >= 4` block 用 mask + compact FP16 side plane を prototype 化し、payload /
   latency を実測して同一 full-model gate を再実行する。
3. `x'_j=x_j/s_j, W'_{ij}=W_{ij}s_j` の per-channel / SmoothQuant calibration を独立に
   prototype 化し、artifact semantics と held-out calibration を明示して再 gate する。
4. 再 gate でも logits/final-hidden max abs <=1.0、top-1 >=99.0%、Wilson >=98.5%、zero
   disallowed mismatch を緩和しない。
