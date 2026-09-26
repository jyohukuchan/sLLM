# Phase 87 段階6: graph内の並列枝（2026-09-21完了）

## 対象と基準

段階5・MTPなし退行修正後の実装 `b261cd7a` を基準とし、独立した投影をcapture時に並列枝へ分ける。
2026-09-21の計画追加コミット `62768b13` は文書だけの変更である。
採用はexact `gfx1030` のNVFP4 gate/up（M=1／3）のforkだけとする。
通常decodeはMTPなし約3.5%、あり約1.7%改善した。FP8 GDNのforkとR9700への適用は棄却した。
以下の最初の表はbaselineであり、最終結果は末尾の比較表に記録する。

通常計測はQwen3.8-27B NVFP4、8192入力／128出力、warmup 1回＋計測3回。
同じmodel lock・runner・MTP設定を使い、通常計測にはprofilerとDAG取得interposerを付けない。

| GPU | MTP | 通常 decode token/s（median） | 通常 TPOT ms | profile TPOT ms | kernel node/replay |
| --- | --- | ---: | ---: | ---: | ---: |
| gfx1030 | off | 16.146946 | 61.931214 | 65.150525 | 1170 |
| gfx1030 | on | 30.176041 | 33.138874 | 34.765965 | 1308 |
| gfx1201 | off | 21.482073 | 46.550442 | 50.282711 | 1170 |
| gfx1201 | on | 35.586296 | 28.100705 | 29.959273 | 1308 |

4条件とも、前回の最終計測との生成token列一致、公開KV/GDN長、request解放後のメモリ解放を確認した。
これは状態plane全体の等価性を証明するものではない。通常計測は既存runnerの成功判定に加え、
`check-final-reports.py` で照合した。

profileの時刻は通常TPOTと別の測定領域として扱う。DAG取得はgraph instantiate時だけの読取りであり、
起動時のCPU負荷に影響する。critical pathと改善上限は同じprofile内で算出し、通常TPOTから差し引かない。

## 構造と候補

取得した4条件の現行graphは全て直列。MTPなしは1172 node（kernel 1170）、
MTPありは1325 node（kernel 1308）である。以前の1218 kernel/tokenの値は退行修正前の履歴値。

候補は56層のNVFP4 MLP gate/upと48層のFP8 GDN QKV/Z。共有quantizerの後では入力・scaleを読むだけで、
各memberの出力は別領域である。M1/M3以外の共有scratchを使い得るvariantは候補実装の対象にしない。
captureの同一queue上でquantizerから分岐し、両memberを合流させてから後続へ進む。
queue追加、replay中のhost確保、再instantiate、環境変数追加は行わない。

## 証拠の所在

生成物は `.local-artifacts/phase87/stage6/` に保持し、Gitには含めない。
`baseline-summary.json` は通常計測・profile内訳への参照とSHA-256、
`baseline-binaries.json` は凍結binaryのSHA-256を記録する。
`baseline-<target>-<off|on>-prior-n0.json` は前回との照合結果。
単体probeの初期試行は、最適化不足と冷間校正の不一致のため採用根拠から除外している。

計画: [Phase 87](../../../../plans/archive/2026/09/11-20/phase87-qwen38-nvfp4-single-request.md)。

## 基準DAGのcritical pathと上限

単位はms／decode遷移（128出力のうち127遷移）。graph再生だけの同一profile時刻領域で比較する。
既知のGDN依存まで緩めた範囲の上限で、未知の依存まで除いた全モデルの上限ではない。
copy/fillの実行時間も保存してCPへ加えるため、kernelだけの理想値より保守的な値になる。

| GPU・MTP | graph span | 現行graph内gap | 候補CP kernel | 候補CP copy/fill | 候補CP gap | 上限 | 半分の打切り線 | 最大幅 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gfx1030-off | 64.3476 | 7.2603 | 45.8640 | 0.0030 | 5.9577 | 12.5229 | 6.2615 | 4 |
| gfx1030-on | 33.6590 | 3.2183 | 25.8053 | 0.0088 | 2.6345 | 5.2104 | 2.6052 | 4 |
| gfx1201-off | 49.5976 | 5.2161 | 35.4981 | 0.0024 | 4.3096 | 9.7874 | 4.8937 | 4 |
| gfx1201-on | 28.9560 | 2.4163 | 22.5070 | 0.0081 | 2.0555 | 4.3854 | 2.1927 | 4 |

観測DAGの最大幅は全て1。実装候補の二member packだけなら最大幅2、GDNのB/Aも独立枝にする
既知依存全体なら最大幅4となる。levelの最大node数3と最大反鎖幅4を区別し、chain・diamond・
不均等な枝長の小グラフで算術を確認した。枝別costとpair-only上限は`weighted/final-compact.json`へ記録した。

traceには直列辺でも負のgapが観測され、最大はV620 off 37.838 µs、on 20.932 µs、
R9700 off 21.066 µs、on 8.398 µs。生のoverlapは保存し、CPのgap重みは0以上へ丸めた。
このため観測chainのCP合計とspanには小差がある。R9700 MTPありのhipBLASLt module名はDAG取得時に
`<unknown>`であり、上表はdispatch順・chain構造・grid/block対応に基づく推定を含む。
同条件の厳密なsymbol確認済み範囲の上限は2.8850 ms／遷移。

## 単体graph probe

N=256、同じkernel引数・入力・出力配置で、枝数Kとcaptureに使うqueue数Qを変えた。
各variantでinstantiateは1回。probeのgapはkernel内の計時区間の間隔で、dispatchに加えて計時区間外の
prologue／epilogueも含む。固定周期の`wall_clock64`を使い、CPU整数oracle、出力guard、
時間区間とHIP eventの整合を確認した。各GPU48条件がPASS。実測kernel時間は、
V620で1.169–1.264／5.035–5.255／20.034–20.250 µs、
R9700で1.037–1.042／5.036–5.038／20.036–20.037 µs。

表はkernel 1個あたりのgap（µs、順に1／5／20 µs設定）。Q>Kなら使うqueueはKまで、
Q<Kならqueue順の依存が残るため、実際の幅は`min(K,Q)`になる。

| K | Q | 実幅 | V620 gap/node（1／5／20） | R9700 gap/node（1／5／20） |
| ---: | ---: | ---: | --- | --- |
| 1 | 1 | 1 | 5.858／2.338／2.569 | 2.202／1.977／24.993 |
| 1 | 2 | 1 | 5.860／2.630／2.567 | 2.198／1.976／23.527 |
| 1 | 4 | 1 | 5.772／2.462／2.567 | 2.194／1.977／25.088 |
| 1 | 8 | 1 | 5.196／2.584／2.568 | 2.184／1.978／23.665 |
| 2 | 1 | 1 | 4.428／2.427／2.567 | 1.973／1.975／24.447 |
| 2 | 2 | 2 | 2.460／2.345／0.000 | 0.727／0.000／9.551 |
| 2 | 4 | 2 | 2.462／2.339／0.000 | 0.729／0.633／9.383 |
| 2 | 8 | 2 | 2.466／2.349／0.000 | 0.726／0.000／10.171 |
| 4 | 1 | 1 | 5.864／3.905／2.567 | 1.972／1.974／23.700 |
| 4 | 2 | 2 | 4.175／4.043／1.390 | 1.351／1.229／16.626 |
| 4 | 4 | 4 | 0.785／0.126／0.000 | 9.624／7.503／0.139 |
| 4 | 8 | 4 | 0.797／0.121／0.000 | 9.628／6.731／0.312 |
| 8 | 1 | 1 | 5.864／2.409／2.567 | 1.978／1.966／23.780 |
| 8 | 2 | 2 | 5.031／4.758／1.970 | 1.671／1.479／19.705 |
| 8 | 4 | 4 | 1.623／0.951／0.525 | 10.448／7.621／3.346 |
| 8 | 8 | 8 | 0.826／0.141／0.045 | 9.753／5.890／0.054 |

枝を増やせば単調に改善するわけではないが、両GPUでgapの減少を確認したため実投影の検証へ進んだ。
古いprobeは冷間校正、重いchecksum、timestampの複数writerを含むため破棄し、
`*-single-writer-r1`を最終の単体probe証拠とする。実投影のprobeは単一queueのfrontier変更を使う。

## 実投影の同一プロセス比較

NVFP4のK=5120、N=17408、M=1／3をproduction lowp APIで実行した。
A=serial、B=forkとしてABとBAの測定順を各4回反復し、別途branchの記録順も逆転して照合した。
各measured replayで独立E2M1/E4M3演算oracle（BF16 1 ULP以内）、finite、serialとのbit一致を確認した。
全HIP resourceの解放失敗は終了失敗として扱う。性能の負値は有効な不採用証拠として保存する。

| GPU | M | AB gain µs/pair | BA gain µs/pair | 正のround（AB／BA） |
| --- | ---: | ---: | ---: | --- |
| gfx1030 | 1 | 30.540 | 28.400 | 4/4・4/4 |
| gfx1030 | 3 | 17.580 | 17.620 | 4/4・4/4 |
| gfx1201 | 1 | -86.240 | -85.340 | 0/4・0/4 |
| gfx1201 | 3 | -90.000 | -88.400 | 0/4・0/4 |

R9700ではM1/M3とも全roundで退行した。全モデルでも両MTP条件が大きく退行したため、
両packの暫定候補では`gfx1030`だけに並列captureを適用し、R9700は既存の直列依存へ戻した。
その後、下記のN0不一致が見つかったため、この暫定候補は採用していない。

## 両pack候補のN0不一致と切り分け

最初の両target共通候補ではV620のMTPなし／ありとも基準token列と一致したが、
R9700を直列へ戻した次のbinaryでは、V620 MTPなしのwarmup＋3 measuredの全てで
8番目（0-origin）から基準と異なるtoken列になった。同一process内の列は一致し、
V620 MTPありとR9700は基準と一致するため、未確認の原因を一般的な非決定性と断定しない。
この候補はN0不一致により採用を止めた。

3 binary（基準・最初の候補・target限定候補）のGPU `.hip_fatbin` は同一で、
SHA-256は `8f9bcb28b95ecf7d813a922f0137ea9f1728c629e886b3f0f3107e2117d863ad`。
model fingerprint、weight plan、prompt、seed 123、selector設定も一致する。
変更前binaryの別process再計測（warmup 1＋measured 1）では基準token列と一致した。

source監査ではNVFP4 provider 84／94とFP8 provider 82のprequantized入力へのwriteや
通常bufferの物理aliasは見つからなかったが、これは実モデルの不一致を覆す証拠ではない。
FP8 GDNのforkを外したNVFP4限定候補では、通常計測・profileともMTP有無の全runで基準token列と一致した。
両pack候補の不一致の詳細原因は特定していない。source監査だけで安全とみなさず、FP8 forkは採用しない。

単体FP8の初期速度比較にはeager controlが混ざっていたため、その速度値は採用根拠から除外した。
serial graph controlへ修正後も、graph外のHIP eventで挟んだ区間にhost投入待ちが入り得るため、
採用用の単体計測はgraph内のdevice時計marker間で計測する。
ROCmのcaptured eventによるelapsed-time問い合わせはinvalid resource handleとなったので使用しない。
GPU時計版でもFP8 M3には負のroundがあり、全round正の採用条件は満たしていない。

## 最終採用と打切り

V620のNVFP4 gate/upだけをforkする。capture時にquantizerのfrontierを保存し、二memberを
同じfrontierから起動した後、両memberのfrontierを合流させる。M=1／3と既存whole-capture中の
同一queueに限定する。M=2／4／prefill等、FP8 GDN、他targetの依存関係は変更しない。
新しいruntime環境変数、queue、instantiate、replay中のhost確保は追加していない。

採用用単体計測は両GPUともgraph内のdevice時計markerを使用した。
V620はAB／BAの全8 roundで改善した。最も小さいroundでも、M1は14.40 µs/pair、M3は16.04 µs/pair。
56層と実際のreplay数（off 126、on 50）、127 decode遷移を使って通常TPOTへ換算すると、
それぞれ **1.29%／1.07%** であり、全roundで方向が揃い、1%基準を満たす。
独立oracle、finite、各replayのserialとのbit一致、cleanupもPASS。
R9700は同じdevice時計方式の全8 roundで負のため、このforkを適用しない。

最終のNV-only DAGは、V620のMTP有無で各56 fork・幅2、辺の削除56／追加112。
その他の辺はbaselineと完全一致する。kernel node数は1170／1308のまま。
R9700の幅は1で、MTP有無とも全ての辺がbaselineと一致する。
これにより、KV／GDN状態公開、selector、停止・予算判定、後続処理への依存順を保持する。
今回の通常・profileの128出力ではdiscarded replayは0であり、replay中のinstantiate／host確保も0。

| GPU | MTP | baseline token/s | 採用経路 token/s | 差 | token列 |
| --- | --- | ---: | ---: | ---: | --- |
| gfx1030 | off | 16.146946 | 16.708786 | +3.480% | 一致 |
| gfx1030 | on | 30.176041 | 30.695941 | +1.723% | 一致 |
| gfx1201 | off | 21.482073 | 21.451289 | -0.143% | 一致 |
| gfx1201 | on | 35.586296 | 35.629441 | +0.121% | 一致 |

8192入力／128出力、warmup 1＋measured 3、model／seed／sampling等は同じ。
R9700の差は計画で認めている全モデル計測の約0.5%の変動内で、forkは無効。
R9700の最終NV-only sourceでは先行するgfx1030条件がfalseになり、測定済みの直列経路と同じ操作列になる。
R9700の測定済みbinaryと最終buildのGPU code objectも同一
（`.hip_fatbin` SHA-256 `68a067103fe3efd5b7306678325038f61f4056acf78d89641d488a77cbbf6eb4`）。

### 同じprofiler設定での比較

使用中のrocprofiler-sdk 1.3.2（git `2b22ab0195cc1461cd9abf3b969e9dd7c10af350`）では、
NV-only graphのprofileがcompletion signal待ちで停止した。進行停止を確認して終了した実行はFAILとして保持する。
同じrevision・同じ警告の[上流報告](https://github.com/vllm-project/vllm/issues/54087)を参考に、
既存SDK設定 `ROCPROFILER_QUEUE_INTERPOSITION=0` をprofiling subprocessにだけ設定した。
小型実機testで3072 kernel trace、512 graph launchとの対応、112組のkernel重なり、整数oracle一致を確認した。
V620のbaselineも同設定で再取得し、設定の異なるprofileを差分計算に混ぜていない。
R9700のbaseline／最終経路はどちらもSDK既定設定である。

単位はms／decode遷移。profile TPOTは通常TPOTと別の観測領域である。

| GPU・MTP | profile TPOT 前→後 | intra-graph gap 前→後 | inter-graph gap 前→後 |
| --- | --- | --- | --- |
| gfx1030-off | 65.4361 → 66.6685 | 7.4210 → 6.0884 | 0.0242 → 0.0226 |
| gfx1030-on | 34.7514 → 34.4028 | 3.2291 → 2.3519 | 0.0099 → 0.0095 |
| gfx1201-off | 50.2827 → 50.3177 | 5.2138 → 5.1924 | 0.0371 → 0.0348 |
| gfx1201-on | 29.9593 → 29.9645 | 2.4162 → 2.4414 | 0.0127 → 0.0121 |

V620はgraph内gapがoff約1.33、on約0.88 ms／遷移減った。
offではprofileのGPU busy unionが57.2919→59.6682 ms／遷移へ増え、profile TPOT自体は悪化した。
これは通常計測の改善とは区別して記録する。profileのkernel時間を通常TPOTから引いたり、
profiling中の結果だけで通常性能を主張したりしない。

同じqueue0 baselineから算出したNV-only理想上限はoff 7.4935、on 7.3231 ms／replay。
半分の打切り線は3.7467／3.6616 ms／replay（decode遷移へ換算すると約3.7172／1.4416 ms）。
実測短縮はこの線へ届かない。kernelを並列にした際の時間が直列時と同じまま全て重なる、という
理想化を実測は満たしていない。資源別の原因までは断定しない。
計画の規則に従って探索を終了し、1%採用基準を満たしたV620 NVFP4限定候補だけを残す。
後続の性能候補はkernel融合であり、この作業では開始しない。

## 検証と証拠

- 両targetのproduction HIP release build PASS。
- 通常4条件と最終profile4条件でHIP実行・fallbackなし、非zero replay、token列一致、公開状態長とcleanupを確認。
- V620は通常計測とprofileを別processで実行してN0を確認。FP8を含む棄却候補の出力を最終証拠に混ぜない。
- graph DAGの全辺比較、幅、fork数、async readback、replay中のinstantiate／host allocationを確認。
- C++形式、cargo fmt、workspace全target clippy（warnings禁止）、関連CI 51 tests＋548 subtests PASS。
- hip-runtime→rmsnorm-h3のhash参照を更新し、JSON manifestとRMSNorm H3 contract検査PASS。
- source integration reviewは完了。後の変更はFP8とR9700のfork適用除外で、scopeを拡大していない。

採用用unitは `pack-probe/result-<target>-device-stamps-r1.json`、通常計測はV620
`nv-only-gfx1030-r1`／R9700 `final-gfx1201-r1`。
V620の最終profileは `queue0-{baseline,nv-only}-profile-gfx1030-{off,on}-r1`、
R9700は `profile-gfx1201-{off,on}-r1`／`final-profile-gfx1201-{off,on}-r1`。
`weighted/matched-baseline-gfx1030-{off,on}-compact.json` と
`weighted/candidate-v-nv-only{,-on}-dag.json` が最終のDAG証拠。
生成物・binary・raw traceは全て `.local-artifacts/phase87/stage6/` に置き、Gitには追加しない。

計画: [Phase 87](../../../../plans/archive/2026/09/11-20/phase87-qwen38-nvfp4-single-request.md)。

完了監査: `completion-audit.json`（27証拠、SHA-256 `7eb5924f999055eb5b51ab6fd859ee24524426aa64fbeb334de0e489fd8fb0f7`）。
最終source SHA-256: `f8cee126acc1c921822cf85c18fe5f851c63ef70b2a686c6214512669c302dd7`。


## 追加再評価

再現しないFP8不一致はユーザー判断で採否根拠から除外し、FP8 M1だけを追加採用した。
NVFP4-only比の通常MTPなしは+0.76%、MTPありは同等、固定128位置のBF16比KLD差は0。
新たに観測したHIP signal underflow停止とtarget別scheduler設定を含め、
[調査・再評価履歴](phase87-fp8-fork-investigation.md)を現在の採否の正本とする。
上記の初回計測値・棄却経緯・source identityは初回完了時点の履歴として保持する。
