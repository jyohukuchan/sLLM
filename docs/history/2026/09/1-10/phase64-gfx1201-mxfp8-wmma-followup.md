# Phase 64 gfx1201 MXFP8 WMMA follow-up履歴

## 2026-09-01: 4候補の実装と限定採用

- Phase 63 ID31を固定baselineとし、exact gfx1201専用のID32 4-wave、ID33 stride-33 LDS pad、ID34 direct-weightを
  additiveな公開kernel ID／symbolとして実装した。ID32だけworkgroup 128、他は256であり、dispatch evidenceもvariant固有の
  workgroup sizeを返すようにした。
- ID33は公開rocWMMA fragment loaderに任意XOR addressingがないため、private layoutへ依存しないstride-33 paddingでbank mappingを
  摂動する候補とした。ID34はweight valueだけをglobalからfragmentへ直接読み、activation valueとE8M0 scaleのLDS共有、
  FP32 accumulation、BF16 RNEを維持した。
- 最終operator runner `sha256:9d3ebadcc0fbb70940c31c3eefaa7beea2cbafab445a937950384343670321cb`で
  5反復した。4B wideのID31／ID32／ID33／ID34中央値は`554,517/671,158/787,196/368,599 ns`、
  9B wideは`1,256,112/1,816,232/1,504,874/866,516 ns`だった。4-waveとLDS padは棄却、direct-weightは
  wideで31.02〜33.53%短縮した。
- direct-weightは2B wideで7.90%、9B down K12288/N4096で33.10%短縮した。一方K11264/N4096は16.08%悪化し、
  4B down K9216/N2560は独立sweep間で符号が変わったため、down family全体へ一般化しなかった。
- ISA/resourceはID31のLDS/SGPR/VGPR `6912/33/103`に対し、ID34は`4864/40/93`、private/spill 0、static WMMA 8だった。
  B-value LDS stagingを除去しながらresident weight repack、GGUF変更、persistent workspaceを追加していない。
- selectorはmodel名非依存で、exact gfx1201のPhase 63 scope内にあるinteger `N/K >= 3`、またはexact
  K12288/N4096だけをID34へrouteする。隣接K11264/N4096と4B downはID31 rollbackを維持した。
- 最終CLI `sha256:7da63d769c2a5d23056bec8e3c4e7abd4be3373b6d21e80756391e681e303a7c`の2,048-token
  prefillは、4Bが`1,745.981 -> 2,215.751 tok/s`（+26.91%）、9Bが`907.962 -> 1,290.812 tok/s`
  （+42.17%）だった。生成token、resident／peak、HIP-only、fallback、cleanupはprovider間で不変だった。
- exact gfx1201 oracleと実モデル、host selector、gfx1030／gfx942 release compile-only、Rust focused test、format、JSON parseを
  PASSし、Phase 64を完了した。追跡済み正本は
  [`phase64-gfx1201-mxfp8-direct-weight-v1.json`](../../../../../ci/matrix/phase64-gfx1201-mxfp8-direct-weight-v1.json)である。

[全体計画](../../../../plans/main-plan.md) /
[対応する計画](../../../../plans/archive/2026/09/1-10/phase64-gfx1201-mxfp8-wmma-followup.md)
