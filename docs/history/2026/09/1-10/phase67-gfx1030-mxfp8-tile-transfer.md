# Phase 67: gfx1030 MXFP8 tile転用とscoped default

2026-09-01にcanonical Radeon Pro V620、exact `gfx1030`、ROCm 7.14.0、Code Object V6、wave32で完了した。
対象はOCP MXFP8 E4M3 W8A8 block32/E8M0、FP32 accumulation、BF16 RNE outputである。ID37やrocWMMAは
gfx1030へ送らず、software E4M3 decodeを使う既存staged MMQ familyのoutput-column再利用だけを評価した。

## 実装と候補

- 既存ID27 col8をcontrolとし、同じ演算順を16列／32列へ広げたexact gfx1030専用ID38
  `matmul.mxfp8.w8a8.gfx1030.mmq-col16.v1`とID39 `mmq-col32.v1`を追加した。両者は
  `SLLM_MXFP8_PREFILL_FORCE_MMQ_GFX1030_COLUMNS=16|32`の明示benchmark overrideだけで選べる。
- K=`31/32/33`、M=1、N tail、別target、override優先順位、prepare-time freezeをhost testへ追加した。
- weight／scale direct-load案は、現在のworkgroupが同じweight tileを8 row-waveで共有するのに対し、直接化するとsoftware
  E4M3 decodeまでwaveごとに反復するため、実装前の構造分析で棄却した。永続BF16／FP32 weight planeは追加していない。
- code object resourceはID27/38/39の順にLDS `8,704/17,152/34,048` byte、SGPR `29/30/30`、
  VGPR `46/42/83`、private 0、spill 0、wave32だった。

## operator結果

row8 ID22、既存col8 ID27、col16 ID38、col32 ID39を18 case、各10回で比較した。全providerのBF16 output digestは
caseごとに一致し、HIP-only、fallback false、cleanup 0だった。代表的な中央値は次の通り（ns）。

| shape | ID22 row8 | ID27 col8 | ID38 col16 | ID39 col32 |
| --- | ---: | ---: | ---: | ---: |
| M128/K2560/N9216 | 10,998,829 | 3,644,210.5 | 3,704,872.5 | 5,946,292 |
| M128/K9216/N2560 | 10,858,668.5 | 3,593,330.5 | 3,750,353 | 5,753,710 |
| M512/K2560/N9216 | 47,151,148 | 20,073,311.5 | 19,987,532.5 | 24,213,592 |
| M512/K9216/N2560 | 43,485,081.5 | 14,137,921 | 14,493,326.5 | 22,570,976 |
| M17/K2560/N32 | 114,401 | 191,142 | 384,324.5 | 528,965.5 |
| M128/K2560/N1024 | 1,179,870.5 | 1,945,577 | 2,049,698 | 2,190,389.5 |

ID38は一部shapeでID27と同等または僅かに速いが、wide/down全体や実モデルで一貫して勝たない。ID39はVGPR/LDS増加と
並列度低下が大きく、短M wide以外では不利だった。従ってID38/39は明示benchmark-onlyに留める。

一方、既存ID27はM>=128の大きなprojectionでrow8を57〜67%短縮した。N=1024はM=128では64.9%悪化するが、
M=512では`4,973,783.5 -> 1,881,010.5 ns`（62.18%短縮）、M=2048では
`19,786,690 -> 9,651,305 ns`（51.22%短縮）へ交差した。この結果から、exact gfx1030かつ
`M>=128, K>=2048, K%32=0`で、`2560<=N<=16384`または`M>=512 && N==1024`だけID27をscoped defaultとした。
短M、M<512のN=1024、未計測N、語彙head、K境界、別targetはrow8を維持する。rollbackは
`SLLM_MXFP8_PREFILL_FORCE_ROW8=1`である。

## Qwen3.5-4B full-model

固定MXFP8 GGUF、FP16 KV、direct pretokenized input、最大4 output、1 warmup＋3 measuredを、最終CLI
`sha256:419c6d3745bc60a763922f5e3517ae81fc17fd0f1031477d7cb1f2787028f787`上で比較した。

| input | row8 median | scoped default median | speedup | prefill時間短縮 |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 72.1830 tok/s | 207.6111 tok/s | 2.8762x | 65.23% |
| 2,048 | 71.2428 tok/s | 208.2710 tok/s | 2.9234x | 65.79% |

residentは両長とも`4,954,035,712` byte、peakは512で`5,292,664,320`、2,048で`6,153,623,040` byteであり、
provider間で増えていない。全sampleの生成tokenは`[23066,23066,23066,23066]`、HIP-only、fallback false、cleanup 0だった。
512-token rocprofv3ではID27 device symbolを800 dispatch、kernel time 92.04%で確認し、ID38/39は0 dispatchだった。

## 結論と境界

gfx1201 ID37の「N方向の再利用を増やす」考え方自体はgfx1030にも有効だが、16/32列へさらに広げる転用は既存8列を
上回らなかった。今回の主な改善は、過去に短caseだけでbenchmark-onlyだった既存ID27について、長prefillとshape crossoverを
追加測定して安全なmodel非依存selectorを確定したことである。量子化recipe、KV default、decode M=1、sampling、public ABIは
変更していない。このevidenceをgfx1031–gfx1036、gfx1201、gfx942、別model、別ROCm tuple、複数GPUへ一般化しない。

[全体計画](../../../../plans/main-plan.md) /
[対応する計画](../../../../plans/archive/2026/09/1-10/phase67-gfx1030-mxfp8-tile-transfer.md) /
[追跡要約](../../../../../ci/matrix/phase67-gfx1030-mxfp8-tile-transfer-v1.json)
