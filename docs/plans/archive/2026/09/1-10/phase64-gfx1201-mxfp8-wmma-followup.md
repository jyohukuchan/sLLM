# Phase 64: gfx1201 MXFP8 WMMA follow-up（完了）

## 目的

Phase 63でexact `gfx1201`へ限定採用したOCP MXFP8 E4M3 W8A8 prefill WMMA経路について、
Qwen3.5-4B／9Bのmodel名に依存しないshape-based providerのまま、残るkernel内最適化候補を独立評価する。

## 固定baseline

- kernel ID 31 `matmul.mxfp8.w8a8.e4m3.block32.prefill.wmma128x64x32.v2`。
- exact R9700 `gfx1201`、ROCm 7.14.0、Code Object V6、wave32、FP32 accumulation、BF16 RNE output。
- Phase 63と同じrow8 rollback、operator oracle、Qwen3.5-4B／9B MXFP8 GGUF、FP16 KVを使用する。

## 評価した4候補

1. 8-wave `M128 x N64`に対する4-wave／workgroup 128候補（kernel ID 32）。
2. A/B LDS tileのstrideを32から33へ変えるbank-mapping候補（kernel ID 33）。公開rocWMMA loaderは任意の
   XOR-addressed fragment loadを公開していないため、private fragment layoutへ依存しないpadding版をXOR候補の代理として評価した。
3. weight valueのLDS stagingを除きglobalからfragmentへ直接loadする候補（kernel ID 34）。E8M0 scaleの共有は維持し、
   stored weight layout、GGUF、loader、resident weightは変更しない。
4. 勝者を`target + dtype/encoding + M/K/N + layout`で選ぶmodel名非依存shape selector。

## 固定した受入条件

- 各kernel候補はbaselineと同じ入力を使い、非整列Mとshape境界を含むoperator oracleでfinite分類、相対誤差、
  BF16 output digest、actual kernel ID／完全symbol／workgroup、fallbackなし、cleanup 0を確認する。
- 性能は同一runner identity内で5反復し、wide、down、small-Nを分離する。単一shapeの勝利を全shapeへ一般化しない。
- production selector候補は同一CLI binaryのQwen3.5-4Bと9B、2,048 input、FP16 KV、1 warmup＋3 measuredでも比較する。
- attention、KV型、FP32 state、公開API意味論、gfx1030／gfx942経路は変更しない。

## operator結果

同一最終runner `sha256:9d3ebadcc0fbb70940c31c3eefaa7beea2cbafab445a937950384343670321cb`の
5反復中央値を採用した。deltaは小さいほど良い。

| shape | ID31 baseline | ID32 4-wave | ID33 LDS pad | ID34 direct weight |
|---|---:|---:|---:|---:|
| M128/K2560/N9216 | 554,517 ns | 671,158 ns（+21.03%） | 787,196 ns（+41.96%） | 368,599 ns（-33.53%） |
| M128/K4096/N12288 | 1,256,112 ns | 1,816,232 ns（+44.59%） | 1,504,874 ns（+19.80%） | 866,516 ns（-31.02%） |

direct-weightの追加境界は、2B wide K2048/N6144が`522,517 -> 481,239 ns`（-7.90%）、9B down
K12288/N4096が`2,150,986 -> 1,438,954 ns`（-33.10%）だった。一方、隣接K11264/N4096は
`2,035,867 -> 2,363,190 ns`（+16.08%）へ悪化した。4B down K9216/N2560は最終runで-3.93%だったが、
独立sweep間で利得の符号が変わったため採用範囲へ一般化しない。

全候補でrepeat output digestは一致し、nonfinite mismatch 0、fallback false、cleanup 0だった。wide 4Bの最大相対誤差は
`0.0036960265`、wide 9Bは`0.0033095016`、9B downは`0.0019967374`で、相対誤差上限`0.02`内だった。

## resourceと採否

| kernel | LDS | SGPR | VGPR | private/spill | static WMMA |
|---|---:|---:|---:|---:|---:|
| ID31 baseline | 6,912 byte | 33 | 103 | 0 | 8 |
| ID32 4-wave | 4,608 byte | 32 | 103 | 0 | 8 |
| ID33 LDS pad | 7,104 byte | 33 | 104 | 0 | 8 |
| ID34 direct weight | 4,864 byte | 40 | 93 | 0 | 8 |

- ID32はLDSを減らしたが、wide 2形状とも遅いため棄却した。
- ID33は公開API内で可能なbank-mapping probeとして棄却した。専用private/custom loaderによる真のXORは未実装だが、
  padding候補が悪化した現時点では追加loaderの複雑性を正当化する利得根拠がない。
- ID34はB-value LDS 2,048 byteを除去し、VGPRも103から93へ減った。stored weight preshuffleを追加せず目的を達成したため、
  resident layout変更は行わない。

## model非依存selectorと実モデル

ID34はexact `gfx1201`、Phase 63 scope（M>=128、K>=2,048、N=1,024〜16,384、K%32=0、N%64=0）のうち、
integer `N/K >= 3`のwide family、またはexact K=12,288/N=4,096に限定して既定採用した。その他はID31へ戻す。
Qwen3.5-4B／9Bというmodel名やrevisionはrouting keyに含めない。

同一CLI `sha256:7da63d769c2a5d23056bec8e3c4e7abd4be3373b6d21e80756391e681e303a7c`で、
2,048 input、chunk 2,048、最大4 output tokenを測定した。

| model | 強制ID31 | selector | 改善 | resident / peak |
|---|---:|---:|---:|---:|
| Qwen3.5-4B MXFP8 | 1,745.981 tok/s | 2,215.751 tok/s | +26.91% | 4,954,035,712 / 6,153,623,040 byte |
| Qwen3.5-9B MXFP8 | 907.962 tok/s | 1,290.812 tok/s | +42.17% | 11,205,394,944 / 12,713,263,616 byte |

全runの生成tokenは`[23066,23066,23066,23066]`で一致し、HIP-only、fallback false、cleanup 0だった。
resident／peakはprovider間で不変で、persistent BF16/FP32 weight、FP32 attention、追加workspaceは導入していない。

## 検証と完了判定

- exact gfx1201 operator oracle、4B／9B full-model prefill、public runtime host selectorをPASSした。
- gfx1030 wave32／gfx942 wave64のreal HIP release compile-onlyをPASSし、ID34を他targetへroutingしていない。
- Rust evidence test 3/3、sllm-hip-sys test、cargo fmt、JSON parseをPASSした。
- ID34を`scoped-default`、ID32／ID33を`rejected`、未測定または非単調なdown形状をID31 rollbackとしてPhase 64を完了する。

[全体計画](../../../../main-plan.md) /
[対応する履歴](../../../../../history/2026/09/1-10/phase64-gfx1201-mxfp8-wmma-followup.md) /
[追跡済み要約](../../../../../../ci/matrix/phase64-gfx1201-mxfp8-direct-weight-v1.json)
