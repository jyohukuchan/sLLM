# Phase 67: gfx1030 MXFP8 tile／direct-load転用

状態: `完了・ID27 shape限定採用／ID38-39 benchmark-only`

## 目的と固定scope

Phase 66のgfx1201 ID37で有効だった複数output列へのactivation再利用を、native FP8 matrix命令を持たないexact gfx1030の
software decode＋FP32 accumulation経路へ転用して採否を決めた。canonical Radeon Pro V620、exact `gfx1030`、ROCm 7.14.0、
Code Object V6、wave32、OCP MXFP8 E4M3 W8A8 block32/E8M0、BF16 RNE outputを固定scopeとした。
full-modelはQwen3.5-4B MXFP8、FP16 KV、direct input、最大4 outputである。

## 完了条件と結果

- [x] exact gfx1030専用ID38 col16／ID39 col32、明示override、logical/device identityを追加した。
- [x] K=`31/32/33`、M/N境界、tail、target非選択、override優先順位、prepare-time freezeをhost testでPASSした。
- [x] row8／col8／col16／col32を18 case×10回で比較し、全providerのBF16 output digest一致、HIP-only、fallback false、cleanup 0を確認した。
- [x] ID38/39のresourceとfull-modelを評価し、既存ID27 col8を一貫して上回らないためbenchmark-onlyとした。
- [x] N=1024のM=`128/512/2048` crossoverを追加測定し、exact gfx1030の測定済みshapeだけID27 col8をscoped default採用した。
- [x] 同一最終binaryでQwen3.5-4Bの512／2,048 inputをrow8 rollbackと比較した。
- [x] history、main plan、GPU／software互換性、数値変更台帳、追跡JSONを同期した。

## 最終selector

OCP MXFP8 prefill、exact gfx1030、`M>=128, K>=2048, K%32=0`で、`2560<=N<=16384`または
`M>=512 && N==1024`なら既存ID27 col8を選ぶ。M=1、短M、M<512のN=1024、未計測N、語彙head、K境界、別targetは
既存row8を維持する。rollbackは`SLLM_MXFP8_PREFILL_FORCE_ROW8=1`である。

## 最終性能

同一CLI `sha256:419c6d3745bc60a763922f5e3517ae81fc17fd0f1031477d7cb1f2787028f787`、
1 warmup＋3 measuredの中央値は次の通り。

| input | row8 | scoped default | speedup | prefill時間短縮 |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 72.1830 tok/s | 207.6111 tok/s | 2.8762x | 65.23% |
| 2,048 | 71.2428 tok/s | 208.2710 tok/s | 2.9234x | 65.79% |

生成token、resident／peak、HIP-only、fallback、cleanupはprovider間で一致した。profileはID27を800 dispatch、
kernel time 92.04%で確認した。weight direct-loadは8 row-waveでsoftware E4M3 decodeを反復するため構造上棄却し、
永続BF16／FP32 weight planeは追加していない。

## 境界

model名、layer、prompt、token、case名はselector keyに含めない。KV cache形式・既定値、quality policy、attention、decode M=1、
public ABIは変更しない。結果をgfx1031–gfx1036、gfx1201、gfx942、別model、別ROCm tuple、複数GPUへ一般化しない。

[全体計画](../../../../main-plan.md) /
[matching履歴](../../../../../history/2026/09/1-10/phase67-gfx1030-mxfp8-tile-transfer.md) /
[追跡要約](../../../../../../ci/matrix/phase67-gfx1030-mxfp8-tile-transfer-v1.json)
