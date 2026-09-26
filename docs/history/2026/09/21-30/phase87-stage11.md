# Phase 87 段階11: Paged attention kernel最適化

2026-09-25着手・採否完了。段階10でexact `gfx1030`／`gfx1201`をPaged KVへ統一した後の
attentionを基準にする。候補と採否条件は[Phase 87計画](../../../../plans/archive/2026/09/11-20/phase87-qwen38-nvfp4-single-request.md#段階11-attention-kernelの最適化2026-09-24追加)を正とし、
C1（MTP verify行間のKV共有）、C2（R9700のGQA共有／V620 M1内側loop）、
C3（prefillの行列演算命令化）を別候補として扱う。既定経路の変更は、同一processの
数値・性能対照と共通採否ルールに従う。

## 着手時の測定範囲

- 段階10のモデル速度対照は`129 input / 最大17 output`で、EOSにより実生成5 token・decode transition 4回だった。M1専用、MTP verify M3専用、8192入力prefillのproduction基準には流用しない。
- Stage10後binaryの両GPUで`8192 input / 最大17 output`、MTP幅2、warmup 0・measured 1の基準runを実施した。V620 `gfx1030`のprefill／TTFTは`35,590.942／35,635.785 ms`、R9700 `gfx1201`は`17,981.770／18,019.351 ms`。両方とも実生成17 token、MTP受理8/16、HIP-only、fallbackなし、cleanup0。reportはignored `.local-artifacts/phase87/stage11/baseline/gfx{1030,1201}-paged-8192-17-width2-r1.json`。report SHA-256はV620 `9c38274b4cf431000fc18c4e59d89a485b68a9563de8cf8f5b93891ad3fa466d`、R9700 `b2dfde939f61ed64bbfc30a859c4d6b7352a322e894974fab5ded605c3f67be8`。単回cold値であり、候補採否のAB/BA値ではない。
- WU-P1時代のV620 `gfx1030` decode M1／KV8192 profilerはGPU busy 100%、MemUnitBusy平均78.4%、VGPR72、LDS16,384 byte、occupancy44.1%だった。これは旧probe binaryの参考値であり、段階10後sourceの性能証拠ではない。逆量子化とonline softmaxの個別費用はcounterから分離できていない。現行sourceのdirect probeで再確認する。
- 現行sourceを使うC1 direct probeの既存Paged stage1（M2/3、8191／8192／8193境界）ではGPU busy 100%、MemUnitBusy中央値87.0%、VGPR72、LDS16,896 byte、occupancy中央値55.1%を観測した。M1単独ではなく、KV読み出しと演算の個別費用も分離できないため、V620 M1内側loopの律速を確定した証拠にはしない。
- Magpie CLI/MCPはこの環境に存在しないため、既存HIP probeとROCm profilerを使う。

## C3初期screen: packed FP16 dot2とFP8 WMMA

現行Paged prefillのQK部分について、`M=128`／KV8192、Qwen3.8
`q_heads=24`／`kv_heads=4`／D256、MXFP8 E4 KV・BF16 Qのtest-only
microprobeを実行した。FP32 scalar controlとpacked FP16 dot2候補の
試作sourceはignored `.local-artifacts/phase87/stage11/c3-dot2-probe.hip.cpp`へ保管した。
同一process AB/BA/ABで、中央値はV620 `gfx1030`が`28.2762 → 34.5701 ms`
（候補約22.3%退行）、R9700 `gfx1201`が`21.0055 → 26.5135 ms`
（約26.2%退行）だった。query/keyのFP16変換はV620が
`0.02008／0.098441 ms`、R9700が`0.01772／0.09636 ms`。
独立FP64 oracleのサンプルはcontrol／候補とも最大誤差0、fallback0、cleanup0。
dot2方式は両GPUで退行したため本番へ採用せず、この試作を打ち切る。
これはprefill全体のWMMA／別tile案を測定した結果ではない。

R9700 `gfx1201`ではBF16 Qをrow-scale E4M3へ変換し、MXFP8 E4 Kの
block32 scaleを補正するrocWMMA FP8 QK microprobeも実行した。
scalar FP32 QK `4.847928 ms`に対し変換込みWMMAは`0.147182 ms`
だったが、独立FP64 oracleとのQK最大絶対誤差はscalar
`1.86e-9`に対してWMMA `0.0548356`、候補とscalar間は`0.0679598`。
softmax差の最大は`5.51e-6`でfinite、HIP実行、cleanupはPASSしたが、
品質非増加条件を満たさないためこの候補は不採用とした。試作sourceはignored
`.local-artifacts/phase87/stage11/c3-wmma-probe.hip.cpp`に保管し、
本番へQ量子化・WMMA経路を追加しない。

## C2-A初期screen: gfx1201 GQA共有

R9700 `gfx1201`のPaged MXFP8 E4 decodeで、現行wave-splitと既存GQA-shared
stage1を同一processの直接probeで比較した。127／128／129、
8191／8192／8193の両側、M=1〜5の30条件はbitwise一致、独立oracle、
status0、invalid-ID拒否をPASSした。KV8191／8192／8193のM1/M3に絞った
AB/BA各3 roundでは、sharedの改善率はAB `−0.145%／+0.960%／+0.550%`、
BA `−1.909%／−0.058%／−0.260%`で、全round 1%以上の改善を満たさなかった。
R9700のGQA共有単独は既定へ採用しない。試作sourceはignored
`.local-artifacts/phase87/stage11/c2-gqa-share-probe.hip.cpp`に保管した。
M3でKVを行間共有するC1は別候補であり、この結果から自動的に棄却しない。

## C2-B初期screen: gfx1030 M1内側loop

V620 `gfx1030`で、M=1専用のcompile-time候補をtest-onlyで比較した。
KV8191／8192／8193、65535／65536／65537は既存Pagedとの
bitwise一致・独立oracle、fallback0、cleanup0をPASSした。
1 warmup＋AB/BA各2 roundでは短い境界付近の差が概ね±1%以内で、
KV65536は一部で約8.6%退行した。共通ルールの全round 1%以上の改善を
満たさず、本番へ採用しない。試作sourceはignored
`.local-artifacts/phase87/stage11/c2-m1-probe.hip.cpp`へ移し、
専用CMake test登録も削除した。

## C1候補: M3の行間KV共有

M=3／committed KV8192以上、exact `gfx1030`／`gfx1201`、
MXFP8 E4 GQA6だけを対象とする。M=2と短いKVは既存Pagedを維持する。
同じQK還元・online softmax・value累積helperを既存stage1とC1で共有し、
K/Vの8-token tileを3 query行へ再利用する。公開ABIではkernel ID118と
専用symbolを区別し、eagerとwhole graph captureの双方を接続した。
127／128／129、8191／8192／8193、65535／65536／65537の
M2/M3直接probeは両GPUでbitwise／独立oracle、ControlV1 partial replay、
overflow拒否、fallback0、cleanup0をPASSした。M2のAB/BAには単発退行があり、
採用範囲から除外した。M3のKV8191／8192／8193は各3 warmup後の
AB/BA各3 paired roundすべてでC1が短かった。KV65537のM3も
V620 `5.3515→3.5361 ms`、R9700 `4.5439→2.7573 ms`で短縮。
R9700の性能対照は既存GQA-shared kernelとし、行間共有の効果を
C2-AのGQA共有単独と分離した。M3/P32の`localSizeBytes`は対照に対し
両GPUともthread当たり約400 B増える一方、active blocksはともに1。

production `8192 input / 最大17 output`、MTP幅2、
1 warmup＋2 measuredの結果は次のとおり。単回cold結果はこの表に混ぜない。

| exact GPU | 経路 | TPOT ms | TTFT ms | E2E ms |
| --- | --- | ---: | ---: | ---: |
| V620 `gfx1030` | 旧Paged対照 | 50.722 | 36,315.654 | 37,127.262 |
| V620 `gfx1030` | C1 M3候補 | 49.658 | 36,362.141 | 37,156.739 |
| R9700 `gfx1201` | 旧Paged対照 | 43.508 | 17,523.692 | 18,219.928 |
| R9700 `gfx1201` | C1 M3候補 | 41.630 | 17,470.545 | 18,136.695 |

TPOTはV620で2.10%、R9700で4.32%短縮した。TTFTはV620
+0.128%で対照2 sample内の揺れより小さく、R9700は0.303%短縮。
各runの実生成17 tokenのSHA、MTP受理8/16、HIP-only、fallbackなし、
cleanup0、execution-session high-water `24,824,122,080 B`は
同じGPUの両経路で一致した。reportはignored
`.local-artifacts/phase87/stage11/baseline/gfx{1030,1201}-{control,c1}-8192-17-width2-w1m2.json`。
対照binary SHA-256は[段階10の退役後記録](phase87-stage10-paged-kv.md)を参照。
C1候補binary SHA-256はV620
`1218d7ac5dfc7110fdc37b93c22145602d72c790e617efc82fb46601a80e899f`、
R9700 `e8ae94bbea451dbc293a56b1ff02c0df93c9087f52cd5c8ba38e352ebe607d3c`。
候補binaryには既存Paged stage1とC1で算術を共有するhelper抽出も含まれる。
行間共有だけの時間差は同一sourceの直接AB/BAで切り分け、上表はbinary全体の
モデル指標として扱う。
同じ`8192 input / 最大17 output`の別監視runによるROCm物理VRAM peakは
V620が対照`28,855,906,304 B`／C1`28,855,988,224 B`（+81,920 B）、
R9700が対照`29,016,657,920 B`／C1`29,016,813,568 B`（+155,648 B）。
4 runともreport PASS、同一token SHA、cleanup0、監視sample error0で、
速度中央値へ混ぜない。VRAM増加は256 MB／VRAM 1%の小さい方を下回る。
bitwise N0、direct kernel全round改善、production TPOT 1%以上改善、
TTFTの重大退行なし、VRAM上限内、fallback／cleanup維持を確認し、
C1のM3・KV8192以上を両targetで限定採用する。

統合レビューではcorrectness/security blockerはなく、benchmarkの固定
`selected_kernel_counts`にID118が欠けていたため追加した。whole graphの
exact identity iterationは既存auditで未対応で、report上のcountだけでは
C1実起動を証明できない。追加のV620 kernel traceは、C1 stage1を
48 dispatch（ControlV1 true 32、false 16、grid 98,304 threads）記録した。
trace CSV SHA-256は
`828e855d97396dff5a75ec3b78331c97ef3cc64bceb6c6894ebbb6b1952b2ed3`。
ただしrocprofv3が同一HSA signalの待機を繰り返して要求を完了できず、
profiled processを停止した。この部分traceはkernel名の帰属にだけ使い、
GPU PASSや速度値へ数えない。完了済みの非profiled実モデルrunと
直接GPU oracleをcorrectness／性能証拠とする。

計画: [Phase 87](../../../../plans/archive/2026/09/11-20/phase87-qwen38-nvfp4-single-request.md)。
