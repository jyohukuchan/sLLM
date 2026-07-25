# ISA 事実訂正と `SQ9_0` 互換性方針

## 前回の要点

- 先行評価では `__builtin_amdgcn_sdot4` の codegen に `v_dot4` が見えない観測から、
  gfx1201/RDNA4 に INT8 dot がないかのような推論が入り得た。
- 同じ評価は、V620 M=1 で `SQ9_0` が `SQ8_0` 比 +6.069% に留まり、採算条件 +7.29% を
  満たさないこと、INT8 block-scale が容量・ISA・品質で有利であることを記録していた。
- 当時は性能単体の結論として `SQ9_0` を runtime/artifact/campaign candidate から破棄すると
  記録した。

## 今回の変更点

- GPU を使わず、ROCm 7.2.1 の `/opt/rocm/llvm/bin/llvm-mc` で gfx1030、gfx1100、gfx1201、
  gfx942、gfx950 に INT8 dot/WMMA/MFMA/FP8 matrix mnemonics を直接アセンブルした。
  - `v_dot4_i32_i8` は全五 architecture で受理される可搬 baseline だった。
  - gfx1100/gfx1201 は旧 VOP2 累積 `v_dot4c_i32_i8` を受け付けず、VOP3P
    `v_dot4_i32_iu8` を受け付ける。gfx1201 が INT8 dot を欠くという結論は誤りである。
  - gfx1100 と gfx1201 は `v_wmma_i32_16x16x16_iu8` を持つ。gfx1201 はさらに
    `v_wmma_f32_16x16x16_fp8_fp8` を含む FP8/BF8 WMMA も持つ。
  - gfx942 は INT8/FP8 MFMA、gfx950 はそれらに加えて広い INT8 と `f8f6f4` FP8 MFMA を持つ。
- `__builtin_amdgcn_sdot4` は cross-architecture capability probe ではないと明記した。
  local bare-target compile では gfx1100/gfx1201 が `dot1-insts` feature diagnostic となる一方、
  `__builtin_amdgcn_sudot4` は VOP3P `v_dot4_i32_iu8` を出力した。project の wrapper/分岐で
  scalarize しても、命令不存在の証拠にはならない。
- `SQ9_0` の位置付けを訂正した。将来の packer、reader、validator、quantizer、generic dequant
  kernel、runtime loader、served-model manifest の対応対象とする。ただし `SQ9_0` は推奨形式、
  default、auto-selection、性能 campaign、matrix-instruction tuning の対象にはしない。
- 過去の V620 timing、offline error、static ISA count は変更していない。今回も GPU、service、
  active manifest、candidate、release、activation には触れていない。

詳細な直接アセンブル証拠、正しい operand 構文、architecture 別フォーマット選択は
`docs/reference/amd-low-precision-isa-and-format-selection-rocm7.2.1.md` に保存した。

## 次の行動

1. `SQ9_0` compatibility implementation plan を作成し、format validation、tail、malformed input、
   CPU oracle、runtime fail-closed 条件を先に固定する。
2. `SQ9_0` は generic dequant と explicit manifest selection だけを実装し、gfx1030、gfx1100、
   gfx1201、gfx942、gfx950 で個別 differential が通るまで runtime capability を主張しない。
3. `SQ8_1` 設計側は `v_dot4_i32_i8` を portable baseline にし、gfx1100/gfx1201 の VOP3P dot と
   gfx1201 INT8 WMMA を反映する。別作業中の `docs/plans/sq8_1-format-design-input-v0.1.md` は
   この task では変更しない。
4. gfx1201/RDNA4 の推奨最適化フォーマットは `SQ8_0` のままとする。`SQ8_1` と `SQ9_0` の
   performance claim は、別途明示承認された GPU window と correctness gate が揃うまで行わない。
5. final activation は人間の明示承認が必要であり、この訂正はそれを許可しない。
