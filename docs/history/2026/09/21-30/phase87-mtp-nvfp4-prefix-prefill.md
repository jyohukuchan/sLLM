# Phase 87 WU-3P: NVFP4 MTP prefix prefill改善記録

2026-09-25着手。現行のMTP prefix準備では、fc/q/k/vを最大1024行ずつ計算し、
NVFP4 W4A4の選択が汎用`row8_tiled256`へ落ちる。段階3のprofileでは32起動が
V620約7.47秒、R9700約5.95秒を占めた。対象は[WU-3P計画](../../../../plans/archive/2026/09/21-30/phase87-mtp-nvfp4-prefix-prefill.md)へ固定した。

## 候補の選択と数値

通常8192入力のprefixはM=1、M=1024が7回、M=1023が1回となる。fcはK10240/N5120、
qはK5120/N12288、k/vはK5120/N1024。現行row8はM1024のqだけで
V620約442 ms、R9700約356 msを費やす。現行と既存kernelを同じshape・入力で比較した。

| GPU | M1024 shape | row8 ms | 採用候補 ms | 候補 |
| --- | --- | ---: | ---: | --- |
| V620 | fc | 366.1 | 6.75 | ID62 DP4A 64×64 |
| V620 | q | 442.2 | 6.51 | ID62 DP4A 64×64 |
| V620 | k/v | 36.64 | 4.08 | ID62 DP4A 64×64 |
| R9700 | fc | 296.1 | 6.69 | ID64 WMMA 128×64 |
| R9700 | q | 355.8 | 6.37 | ID64 WMMA 128×64 |
| R9700 | k/v | 31.83 | 0.87 | ID64 WMMA 128×64 |

M1023の末尾chunkも6形状で改善した。M32の境界も、V620のfc/q/k(v)が
17.21→2.58／14.66→1.68／5.54→0.87 ms、R9700が10.56→0.87／12.71→0.65／
1.29→0.88 msで、すべて改善した。row8_col8はqでV620 247.8／R9700 207.6 msにとどまり、
V620の補償DP4Aは7.47 ms、R9700の通常DP4Aは10.70 msで採用候補より遅かった。
新しいdevice算術を増やさず、M32〜1024で上記3つのK/Nに限りgfx1030はID62、gfx1201はID64へ
既定選択した。M1、M31以下、M1025以上、本体MLPのK/Nは従来のselectorを維持する。

独立long-doubleのE2M1/E4M3FN積和oracleを持つ[専用GPU probe](../../../../../native/hip/tests/phase87_wu3p_mtp_prefill_probe.hip.cpp)で、
両GPU各62ケース（M1023/1024の全実形状とM31/32/33、63/64/65、127/128/129、
511/512/513、tiny非整列）を各2 seedでPASSした。各GPUの全BF16出力151,375,970値のうち
現行との差は157値、最大1 ULP、非有限・反復不一致0。独立oracleの744点は候補・現行とも最大0 ULP。
候補のresourceはgfx1030 ID62がVGPR86／LDS6144 B／active blocks 5、gfx1201 ID64が
VGPR97／LDS7684 B／active blocks 7（現行row8はVGPR36／33、LDS1096 B、active blocks 8）。
gfx1030のM32はproduction launcherの別device body ID62 32×64を使い、VGPR59／LDS4608 B／active blocks 8。
probeのPASSは独立oracle差0 ULP、現行との差最大1 ULP、非有限0、repeat差0をすべて検査する。
最終probe binary SHA-256はV620
`22907e3d0fc80eff4e0ad3fa7b43af5339d19262bd2f416c5ff66012f82a1b51`、R9700
`06887f5b7f4ad40f6c878b269b74b322363bd3319c76dc6bf2fd8e63a865c809`。
rawログは追跡対象外の`.local-artifacts/phase87/wu3p/numeric-probe-{gfx1030,gfx1201}-oracle-gated-v2.log`。
ログSHA-256はV620 `e4cecb00acade424940493672bf9ff59ee300be356da35dfffb76cf0e74f21bc`、
R9700 `7f3d84f9ee1cdc33e986c7c0b12724b7db3464a2bd71447b6aa5c12895bb7c7c`。
採用上限の外側M1025/K5120/N12288は、両GPUの公開evidence runnerでID59 row8選択、
独立FP32 sampled oracle、finite、cleanup zeroをPASSした。M31とM1025のselector境界はhostでも確認した。

## 通常8192/128と採否

同じモデル・NVFP4 sidecar・縮小draft head・seedのexact target別binaryで、
各1 warmup＋3 measuredを比較した。各欄はmeasured中央値の秒。MTP prefix wallはprefillに含まれる。

| GPU | 経路 | MTP prefix | prefill | TTFT | decode参考 |
| --- | --- | ---: | ---: | ---: | ---: |
| V620 | 現行NVFP4 | 7.527 | 43.515 | 43.543 | 3.919 |
| V620 | 候補NVFP4 | 0.287 | 36.198 | 36.227 | 3.863 |
| V620 | BF16 companion対照 | 1.572 | 37.628 | 37.657 | 3.909 |
| R9700 | 現行NVFP4 | 6.018 | 20.746 | 20.772 | 3.819 |
| R9700 | 候補NVFP4 | 0.201 | 14.916 | 14.942 | 3.448 |
| R9700 | BF16 companion対照 | 0.110 | 14.832 | 14.858 | 3.368 |

現行NVFP4比でMTP prefixはV620 96.19%、R9700 96.66%短縮、prefillは16.82%／28.10%短縮した。
V620候補は同条件BF16よりprefillで1.430秒短く、R9700候補はBF16より0.084秒長いが
各3回の測定幅内である。NVFP4のresident/peakは候補前後で両GPUとも
22,393,597,824／25,729,515,552 byteに一致した。decode参考値は候補と現行で
生成列・受理数が異なるため、decode kernel単体の改善とは扱わない。

候補の全runはexact GPU UUID、HIP-only、fallbackなし、finite terminal logit、固定K20 p/q、
128出力、決定的なtoken列、request/session cleanup zeroでPASSし、性能levelとR9700 service状態を復元した。
MTPなし8192/128はV620
`sha256:c9c0b4ee401b11544fe0faaec15d882cb2491c31a460a32a89dce28e437bcc0e`、
R9700 `sha256:75d36def8ff45d155373ebb05b885d0e4f0e7197ba9adb2123adfb1fe63b189d`で
段階9対照と一致した。MTPありの最初の出力分岐はV620 108、R9700 53 token目で、
draft演算順の変更を数値台帳にN2候補として記録する。target pとp/q補正規則は不変。

今回実測した現行binary SHA-256はV620
`2b19db0ed888cedbbfbb39120dcd1fca18aca7e952c79cbb66989b26f8a2cc0e`、R9700
`50acc9a733a0344923cca189462e8df1762569028599d776185474cfe6f0f78e`。
候補binary SHA-256はV620
`c0cadfaa5b2e7ccf4016bf307511bb06ed84a50c4099a40500aa524417a8d373`、R9700
`ea20f8db8770d00d9a37646bf3963bd3aa8f874c48466376fea06beb57414eb2`。
runごとのreport hash、GPU UUID、全measured値、cleanupと比較式は追跡対象外の
`.local-artifacts/phase87/wu3p/summary-v1.json`（SHA-256
`b01cb032333fb5b8341483ee1364ce7c079a86f19a091fdf05900df4be5bfa66`）へ保存した。
最終sourceの整形後に両targetを再buildし、候補binaryのfull SHA-256が同じことを確認した。
hostの3,002件selection fixtureと新しいM境界、`clang-format --dry-run --Werror`、
両targetのHIP build、Markdown local linkと`git diff --check`をPASSした。新しい外部source importはない。

計画: [WU-3P](../../../../plans/archive/2026/09/21-30/phase87-mtp-nvfp4-prefix-prefill.md)。
