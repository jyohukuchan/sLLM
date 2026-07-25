# SQ8_1 W8A8 活性誤差実測と設計確定

## 前回の要点

- `SQ9_0` は V620/gfx1030 の M=1 実測と独立した offline 比較の両方から、runtime/artifact/campaign
  candidate として破棄済みだった。
- 後継候補 `SQ8_1` の signed int8 block-scale は、W8A8 に必要な活性側の動的 int8 量子化誤差が未確認であり、
  実モデルでの確認が必要だった。
- 並行セッションによる `docs/plans/sq9-format-design-input-v0.1.md` の V620 実測追記は、元の evidence と照合して
  正しいことを確認し、設計作業とは分離して `e958bb6b` にコミットした。

## 今回の変更点

- 既存の importance-score activation collector (`tools/collect-activation-stats.py`) の loader、corpus parser、
  hook 命名規則を再利用した CPU-only 測定を追加した。Qwen3.5-9B の実 Linear 入力を hook 内で処理し、
  raw activation は保存していない。R9700/gfx1201、V620、HIP/CUDA の実行は行っていない。
- frozen `D_stats-shard-00.jsonl` の 8 record（962 valid token、248 Linear module）で、81,788,928 activation 値を測定した。
  K=32、FP16 scale の `ceil_fp16` 保存では activation relative L2 が 0.00994415、post-storage true clipping は 0、
  sampled W8A8 linear-output relative L2 は 0.00775109 だった。
- 通常の FP16 RNE scale 保存は K=32 で 1.594585% の post-storage clipping を生んだ。scale を最小の
  `>= raw_scale` FP16 値に上向き丸めることで、payload 密度を変えずに clipping を 0 にした。
- K=16/32/64/128 を比較し、K=32 を確定した。FP16-upward activation relative L2 は順に
  0.00725145 / 0.00994415 / 0.01358876 / 0.01832550、full-block density は
  9.0 / 8.5 / 8.25 / 8.125 bpp だった。FP16 は同じ 16-bit storage の BF16 より各 K で低誤差だった。
- K=32 outlier block は `max(abs)/RMS` が [4,8) に 17.1521% 存在し、この strata の median per-tensor
  relative L2 は 0.01163607（[2,4) では 0.00653891）だった。局所的な悪化は確認されたが、観測 block に [8,∞) はなく、
  現標本だけで per-channel/outlier side path を基底 format に入れる根拠はない。
- 1 prompt の activation-only logit smoke は final 16 token で top-1 16/16 一致、relative L2 0.01401899、
  mean KL 0.000323955 だった。重みと活性を同時に量子化した full-model W8A8 logits は未確認である。
- GPU を実行せず device-only compiler recheck を行い、gfx1030 は `v_dot4c_i32_i8`、gfx942 は
  `v_dot4c_i32_i8_e32` を出力した。gfx1201 は `__builtin_amdgcn_sdot4` を missing `dot1-insts` feature として
  拒否した。これは ISA eligibility だけであり、性能実測ではない。
- `docs/plans/sq8_1-format-design-input-v0.1.md` に、K=32、FP16 upward scale、symmetric signed int8、
  separate payload/scale plane、W8A8/W8A16 policy、architecture dispatch rule、実装前の gate を記録した。

## 次の行動

1. `SQ8_1` の bounded-memory CPU reference quantizer を実装し、FP16-upward、tail、RNE、pack/unpack の一致を検証する。
2. W8A16 reference、次に gfx1030/gfx942 向け W8A8 dot4 kernel を実装し、CPU/GPU differential と ISA を確認する。
3. artifact/runtime 採用前に、事前宣言した threshold を用いる held-out full weight-plus-activation model quality gate を実施する。
4. V620 を後続で使用する場合だけ、PCI BDF からそのカード固有の `temp2_input` を継続監視し、junction 85 °C 以上で即中断する。
