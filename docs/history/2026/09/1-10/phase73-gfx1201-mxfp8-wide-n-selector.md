# Phase 73: gfx1201 MXFP8 wide-N selector

## 結論

exact `gfx1201`のMXFP8 production WMMA selector上限を、ユーザー指示により16,384から32,768へ緩和した。
変更はmodel名非依存で、Phase 63～66のID31／34／36／37に共通して適用される。N以外の既存条件は維持する。

## 実装と確認

- `phase63_gfx1201_mxfp8_wmma_shape`、`phase65_gfx1201_mxfp8_wmma_direct_both_shape`、
  `gfx1201_mxfp8_wmma_n64_shape`の上限を32,768へ揃えた。
- N=17,408／32,000／32,768のM128整列shapeはID37 N128 direct-bothを選ぶ。
- N=32,769は列alignmentと上限の双方、N=32,832は上限だけで不採用となり、row8／block32へ戻る。
- public runtime host testは1/1 PASS、exact gfx1201 low-precision codec/provider GPU testはPASSした。

kernel算術、E4M3 value／E8M0 scale、FP32 accumulation、BF16 RNEは変更していない。ただし新しく採用したN範囲の
GPU operator oracle、full-model生成token、性能はユーザー指示に従い未実施である。従ってこれはselector契約の採用であり、
新しいwide-N GPU evidenceや性能保証ではない。rollbackは3 predicateの上限を16,384へ戻す。

[保存済み計画](../../../../plans/archive/2026/09/1-10/phase73-gfx1201-mxfp8-wide-n-selector.md) /
[数値変更台帳](../../../../compatibility/numerical-output-changes.md) /
[追跡要約](../../../../../ci/matrix/phase73-gfx1201-mxfp8-wide-n-selector-v1.json)
