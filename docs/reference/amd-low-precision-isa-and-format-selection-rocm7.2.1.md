# AMD 低精度 ISA とフォーマット選択リファレンス（ROCm 7.2.1、2026-07-26 再検証）

## 目的と適用範囲

この文書は、`gfx1030`、`gfx1100`、`gfx1201`、`gfx942`、`gfx950` の INT8 dot と
INT8/FP8 行列命令を、ROCm 7.2.1 同梱の `llvm-mc` で**直接**再検証した記録である。
特に、`gfx1201`（RDNA4）が INT8 dot を持たない、という誤った推論を防ぐ。

検証はアセンブラと CPU 上のオフラインコンパイルだけで行った。GPU、HIP runtime API、
カーネル起動、サービス、artifact、candidate、release、active manifest は使用も変更もしていない。
したがって、この文書の `OK` は「この toolchain が指定 ISA 向けにエンコードを出す」を意味し、
数値正しさ、fragment layout、occupancy、性能、または runtime dispatch の証明ではない。

`AQ4_0`、`SQ8_0`、`SQ8_1`、`SQ9_0` は厳密フォーマット名である。`AQ4`、`SQ8`、`SQ9` は
カテゴリ名であり、runtime/artifact の識別子としては使わない。

既存の広い GPU 制約は
[gpu-architecture-capabilities-rocm7.2.1.md](gpu-architecture-capabilities-rocm7.2.1.md) を参照する。
この文書はそのうち低精度 ISA の一次的なローカル再検証記録である。

## 検証環境と再現方法

- 実行日: 2026-07-26
- assembler: `/opt/rocm/llvm/bin/llvm-mc`
- version: `AMD LLVM version 22.0.0git`（ROCm 7.2.1 の local toolchain）
- target triple: `amdgcn-amd-amdhsa`
- 対象: `gfx1030`、`gfx1100`、`gfx1201`、`gfx942`、`gfx950`

各セルは、次の形式でニーモニックを各 `-mcpu` に直接渡した結果である。

```bash
printf '%s\n' 'v_dot4_i32_i8 v0, v1, v2, v3' \
  | /opt/rocm/llvm/bin/llvm-mc \
      -triple=amdgcn-amd-amdhsa -mcpu=gfx1201 -show-encoding
```

`OK` は `-show-encoding` が encoding を出したこと、`—` は同じ入力に対する
`instruction not supported on this GPU` を表す。`invalid` はニーモニックそのものが
ROCm 7.2.1 の assembler で有効ではないことを表す。対象を超える dtype（FP16/BF16/INT4）を
除き、`llvm-mc` の命令文字列に現れる INT8、FP8、BF8、FP8 を選択できる `f8f6f4` の
WMMA/MFMA 系を候補にして直接アセンブルした。

## INT8 dot4 の再検証

| 直接入力したニーモニック | gfx1030 | gfx1100 | gfx1201 | gfx942 | gfx950 |
| --- | --- | --- | --- | --- | --- |
| `v_dot4c_i32_i8 v0, v1, v2` | OK | — | — | OK | OK |
| `v_dot4_i32_i8 v0, v1, v2, v3` | OK | OK* | OK* | OK | OK |
| `v_dot4_i32_iu8 v0, v1, v2, v3` | — | OK | OK | — | — |
| `v_dot4_u32_u8 v0, v1, v2, v3` | OK | OK | OK | OK | OK |

`*` gfx1100 と gfx1201 は入力の `v_dot4_i32_i8` を受理し、出力表示では
`v_dot4_i32_iu8` に正規化した。これは「`v_dot4_i32_i8` を入力できない」または
「INT8 dot がない」という結果ではない。実装では、符号制御を含む実際の codegen を対象
architecture ごとに確認する。

したがって、ソース上の可搬な signed INT8 dot4 基準は **`v_dot4_i32_i8`** である。
gfx1100/gfx1201 の最適化 selector は VOP3P の `v_dot4_i32_iu8` を選べるようにし、
gfx1030/gfx942/gfx950 の `v_dot4c_i32_i8` は、その architecture 専用の累積形として扱う。
単一の旧 builtin や単一の表示名を全 architecture の能力表に一般化してはならない。

## `__builtin_amdgcn_sdot4` の罠

`__builtin_amdgcn_sdot4` は全 architecture 共通の capability probe ではない。最小の device
関数を同じ local compiler で `-target amdgcn-amd-amdhsa -mcpu=<arch> -O3 -S` した結果は次の通り。

| target | `__builtin_amdgcn_sdot4` の結果 | 対応する確認 |
| --- | --- | --- |
| gfx1030 | `v_dot4c_i32_i8` を出力 | VOP2 累積形 |
| gfx1100 | `dot1-insts` feature が必要という compile diagnostic | `__builtin_amdgcn_sudot4(true, a, true, b, c, false)` は `v_dot4_i32_iu8 ... neg_lo:[1,1,0]` を出力 |
| gfx1201 | `dot1-insts` feature が必要という compile diagnostic | 同じ `sudot4` probe が `v_dot4_i32_iu8 ... neg_lo:[1,1,0]` を出力 |
| gfx942 | `v_dot4c_i32_i8_e32` を出力 | VOP2 累積形 |
| gfx950 | `v_dot4c_i32_i8_e32` を出力 | VOP2 累積形 |

既存の HIP source では、unsupported builtin を前処理分岐や wrapper が scalar 演算へ落とす場合が
ある。bare target で diagnostic になる場合も含め、`__builtin_amdgcn_sdot4` 一つの codegen に
`v_dot4` が見えないことは、命令不存在の根拠にならない。RDNA3/RDNA4 では VOP2 累積形ではなく
VOP3P の形を確認する必要がある。

この調査で使った VOP3P 確認入力は次である。

```asm
v_dot4_i32_i8 v0, v1, v2, v3
v_dot4_i32_iu8 v0, v1, v2, v3
```

将来の kernel review では、builtin 名だけでなく、対象 `-mcpu` の HSACO/assembly に期待する
`v_dot4_*`、`v_wmma_*`、または `v_mfma_*` が出ることを確認する。

## INT8/FP8 行列命令の再検証

### 結果一覧

`fp8/bf8` は同じ行に並べた四つの入力組、すなわち `fp8_fp8`、`fp8_bf8`、`bf8_fp8`、
`bf8_bf8` をすべて直接アセンブルした結果を表す。

| 命令 family（すべて直接 `llvm-mc` 済み） | gfx1030 | gfx1100 | gfx1201 | gfx942 | gfx950 |
| --- | --- | --- | --- | --- | --- |
| INT8 WMMA `v_wmma_i32_16x16x16_iu8` | — | OK | OK | — | — |
| FP8/BF8 WMMA `v_wmma_f32_16x16x16_{fp8,bf8}_{fp8,bf8}` | — | — | OK | — | — |
| INT8 MFMA `v_mfma_i32_16x16x32_i8`, `v_mfma_i32_32x32x16_i8` | — | — | — | OK | OK |
| INT8 MFMA 拡張 `v_mfma_i32_16x16x64_i8`, `v_mfma_i32_32x32x32_i8` | — | — | — | — | OK |
| FP8/BF8 MFMA `v_mfma_f32_16x16x32_{fp8,bf8}_{fp8,bf8}`, `v_mfma_f32_32x32x16_{fp8,bf8}_{fp8,bf8}` | — | — | — | OK | OK |
| FP8 selectable MFMA `v_mfma_f32_16x16x128_f8f6f4`, `v_mfma_f32_32x32x64_f8f6f4` | — | — | — | — | OK |
| scale 付き FP8 selectable MFMA `v_mfma_scale_f32_16x16x128_f8f6f4`, `v_mfma_scale_f32_32x32x64_f8f6f4` | — | — | — | — | OK |

結論は次の通りである。

- gfx1030 はこの対象の WMMA/MFMA を持たないが、INT8 dot4 は持つ。
- gfx1100 は INT8 WMMA を持つ。FP8/BF8 WMMA はこの ROCm 7.2.1 の直接検証では持たない。
- gfx1201 は INT8 WMMA と FP8/BF8 WMMA を持つ。したがって RDNA4 で `SQ8_1` の INT8 行列経路は成立し、`SQ8_0` の FP8 行列経路も成立する。
- gfx942 は INT8 と FP8/BF8 の MFMA を持つ。
- gfx950 は gfx942 の base MFMA に加え、より広い INT8 と FP8 selectable MFMA を持つ。

### gfx1100 / gfx1201 の正しい WMMA 構文

次の構文は実際に `-show-encoding` まで成功した。gfx1100 と gfx1201 は同じ K=16 命令でも
source register group の幅が異なる。

```asm
; gfx1100: INT8 WMMA, A/B は各 4 VGPR
v_wmma_i32_16x16x16_iu8 v[0:7], v[8:11], v[12:15], v[0:7]

; gfx1201: INT8 WMMA, A/B は各 2 VGPR
v_wmma_i32_16x16x16_iu8 v[0:7], v[8:9], v[10:11], v[0:7]

; gfx1201: FP8 WMMA（ほかの fp8/bf8 組合せも同じ operand shape）
v_wmma_f32_16x16x16_fp8_fp8 v[0:7], v[8:9], v[10:11], v[0:7]
```

`v_wmma_i32_16x16x32_iu8` は ROCm 7.2.1 のこの assembler では `invalid instruction` だった。
また、命令文字列にある `v_wmma_i32_16x16x64_iu8` は対象五 architecture のいずれでも
`instruction not supported on this GPU` だった。K=32 または K=64 を推測で RDNA3/RDNA4 の
feature 表へ追加してはならない。

### 同型候補の負結果（網羅性の境界）

結果一覧は、成功した命令だけを抜き出した表ではない。INT8/FP8/BF8 行列命令として同じ family の
次の候補も、五つの対象すべてに直接入力して確認した。いずれも成功しなかったため、対応表には
載せていない。これにより、ここでいう「網羅」は対象 dtype の WMMA/MFMA の実在する shape と、
同じ family に見えるが ROCm 7.2.1 で使えない shape の両方を含む。

| 直接検証した候補 | 五 architecture における結果 |
| --- | --- |
| `v_wmma_i32_16x16x32_iu8` | 全て `invalid instruction` |
| `v_wmma_i32_16x16x64_iu8` | 全て `instruction not supported on this GPU` |
| `v_wmma_f32_16x16x64_{fp8,bf8}_{fp8,bf8}` | 四つの input 組合せとも全て unsupported |
| `v_wmma_f32_16x16x128_{fp8,bf8}_{fp8,bf8}` | 四つの input 組合せとも全て unsupported |
| legacy 名 `v_mfma_i32_16x16x16i8`、`v_mfma_i32_32x32x8i8` | 両方とも全て unsupported |

この負結果は ROCm 7.2.1 の assembler/target 組合せに限定される。将来の toolchain で
instruction 定義が増えた場合は、同じ direct probe を再実行して表を更新する。

### gfx942 / gfx950 の正しい MFMA 構文

```asm
; gfx942 と gfx950: base INT8 MFMA
v_mfma_i32_16x16x32_i8 a[0:3], v[4:5], v[6:7], a[0:3]
v_mfma_i32_32x32x16_i8 a[0:15], v[4:5], v[6:7], a[0:15]

; gfx950 の追加 INT8 MFMA
v_mfma_i32_16x16x64_i8 a[0:3], v[4:7], v[8:11], a[0:3]
v_mfma_i32_32x32x32_i8 a[0:15], v[4:7], v[8:11], a[0:15]

; gfx942 と gfx950: base FP8 MFMA
v_mfma_f32_16x16x32_fp8_fp8 a[0:3], v[4:5], v[6:7], a[0:3]
v_mfma_f32_32x32x16_fp8_fp8 a[0:15], v[4:5], v[6:7], a[0:15]

; gfx950: FP8 E4M3/E5M2 を選べる広い MFMA
v_mfma_f32_16x16x128_f8f6f4 a[0:3], v[0:7], v[8:15], a[0:3]
v_mfma_f32_32x32x64_f8f6f4 a[0:15], v[0:7], v[8:15], a[0:15]
v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], v[0:7], v[8:15], a[0:3], v16, v17 op_sel_hi:[0,0,0]
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[0:7], v[8:15], a[0:15], v16, v17 op_sel_hi:[0,0,0]
```

`f8f6f4` の FP8 mode は local CK header の `cbsz`/`blgp` contract で選ぶ。上の assembler
成功は命令の availability を示すだけであり、canonical `SQ8_0` OCP payload をそのまま投入できる
という意味ではない。gfx942 の OCP-to-FNUZ prepack 条件は
[sq8-cdna3-port-plan-v0.1.md](../plans/sq8-cdna3-port-plan-v0.1.md) のまま有効である。

## アーキテクチャ別フォーマット選択規則

この規則は選択ポリシーであり、未実装の reader、kernel、artifact、candidate、release、activation を
承認するものではない。常に manifest の厳密フォーマット名を優先し、reader が別形式を黙って
再量子化または置換してはならない。

| architecture | 推奨される最適化フォーマット | ISA に基づく理由と条件 | 互換・非推奨の扱い |
| --- | --- | --- | --- |
| gfx1030 | `SQ8_1`（実装・品質 gate 完了後） | 可搬基準 `v_dot4_i32_i8` と既存 `v_dot4c_i32_i8` を使える。WMMA/MFMA はない。 | `SQ8_0` は source-preserving/reference の扱いで、native FP8 matrix 最適化を前提にしない。`SQ9_0` は将来の explicit compatibility dequant のみ。`AQ4_0` はその厳密 artifact を要求する model の選択肢である。 |
| gfx1100 | `SQ8_1`（実装・品質 gate 完了後） | VOP3P dot と `v_wmma_i32_16x16x16_iu8` を使える。FP8 WMMA はこの検証ではない。 | `SQ8_0` は generic/reference を許容しても native FP8 matrix の推奨対象ではない。`SQ9_0` は explicit compatibility のみ。`AQ4_0` は strict artifact choice のまま。 |
| gfx1201 | **`SQ8_0`** | source FP8 payload を再量子化せず、`v_wmma_f32_16x16x16_fp8_fp8` を含む FP8/BF8 WMMA がある。実装ごとの数値・性能 gate は別途必要。 | `SQ8_1` は VOP3P dot と INT8 WMMA を使える有効な alternative だが、`SQ8_0` より優先しない。`SQ9_0` は explicit compatibility のみで最適化対象にしない。`AQ4_0` は strict artifact choice のまま。 |
| gfx942 | `SQ8_0`（OCP-to-FNUZ gate と native MFMA gate 完了後） | INT8/FP8 MFMA がある。`SQ8_0` は canonical OCP payload を保持し、FNUZ は private prepack に限る。 | `SQ8_1` は `v_mfma_i32_*` に最適化できる候補だが別設計・品質・実機検証が必要。`SQ9_0` は generic compatibility に限る。`AQ4_0` は strict artifact choice のまま。 |
| gfx950 | `SQ8_0`（format gate と native MFMA gate 完了後） | gfx942 の base FP8 MFMA に加え FP8 selectable MFMA がある。canonical payload compatibility は個別に証明する。 | `SQ8_1` は広い INT8 MFMA を使える候補だが別設計・品質・実機検証が必要。`SQ9_0` は generic compatibility に限る。`AQ4_0` は strict artifact choice のまま。 |

`SQ9_0` は対応する wire format と再定義したが、現在この task は実装しない。対応の正確な範囲は
[sq9-format-design-input-v0.1.md](../plans/sq9-format-design-input-v0.1.md) の compatibility section を
正とする。すべての architecture で将来 explicit に選べる generic dequant path を対象にする一方、
default、auto-selection、matrix-instruction tuning、campaign promotion の対象にはしない。

## `SQ8_1` 設計側への申し送り

別作業中の `docs/plans/sq8_1-format-design-input-v0.1.md` はこの task では変更しない。設計側は
次を反映する必要がある。

1. RDNA4/gfx1201 は INT8 dot を持つ。`__builtin_amdgcn_sdot4` の結果だけから否定しない。
2. 可搬な W8A8 の ISA 基準は `v_dot4_i32_i8` とし、gfx1100/gfx1201 は `sudot4`/VOP3P
   `v_dot4_i32_iu8` を selector に置く。`v_dot4c_i32_i8` を RDNA3/RDNA4 の前提にしない。
3. gfx1201 では `v_wmma_i32_16x16x16_iu8` が使える。dot4 と WMMA のどちらを選ぶかは、fragment
   layout、M shape、品質、実機 performance を比較して決める。
4. `SQ8_0` は gfx1201 で source-preserving FP8 WMMA route を持つため、INT8 route が成立することは
   `SQ8_0` の推奨を覆す根拠ではない。

## 継続時の確認事項

1. 新しい HIP kernel は、target ごとの direct `llvm-mc` probe と実際の generated ISA の両方を残す。
2. `SQ8_1`、`SQ8_0`、`SQ9_0` の実装は CPU oracle、artifact validation、各 target の実機 differential
   を通過するまで runtime selection を有効にしない。
3. final activation は別の人間による明示承認が必要であり、この ISA 検証からは導かれない。
