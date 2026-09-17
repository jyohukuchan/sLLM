# Phase85 follow-up: M=1 A16によるMTP高速化の実測

状態: 完了。Stage 0〜3の実験・採否記録、撤去後検証、区間別priming確認を完了した。
BF16 companion既定は変更していない。

## 対象と判定

Qwen3.8-27B NVFP4本体、MTP専用8行列の既存MXFP8／MXFP6 sidecarを使う。
共有BF16 embedding／FP8 head、norm、KV encodingはMTP重み量子化の対象外。
M=1だけBF16 activationを維持し、M>1では既存A8／A6とWMMA選択を使う。

Stage 0は両GPUの12言語・タスク条件×3 seed、各128行、5系列を比較した。
同一GPUではBF16生成prefixとtarget hiddenをhashで固定し、raw logitsのargmax／top5／相対L2を再計算した。
CPUのfake-quant往復は各形式8行列424,673,280要素すべてでBF16 bit一致、
非finite・overflow・underflow・bit不一致はいずれも0だった。

| GPU | MXFP8 A8のBF16 top1一致 | 重みのみMXFP8の一致 | BF16との差の回復率 | 判定 |
| --- | ---: | ---: | ---: | --- |
| V620 | 4501 / 4608 | 4536 / 4608 | 32.71% | 両方が寄与、Stage 1へ |
| R9700 | 4507 / 4608 | 4531 / 4608 | 23.76% | 活性化の寄与は小さい |

合算は59/208＝28.37%。計画で集約方法が未指定だったため当初GPU別に判断し、
2026-09-14のユーザー回答でGPU別判定が明示決定された。V620の条件成立を根拠に共通実装を両GPUで検証する。
通常MTPのcoding-en／creative-ja、seed123、128出力も5系列・両GPUで測定した。
こちらは生成履歴が分岐するため、固定prefixの主指標とは区別する。

## Stage 1の実装と演算子検証

`SLLM_MX_WA_M1_A16=1`をprepare時に読む。nativeへは既にBF16 activationが渡っていたため、
Rust graph側に量子化分岐を増やさずnative内部のquantizerとworkspaceを省いた。
新ID101／102はM=1だけに適用する。force-baselineは従来のID18／20を優先する。
重みcodec、Kのlane割当、FP32積和、reduction順はColumns2から維持した。
A16の数値はA8／A6と異なるため、前後の出力digest一致を受入条件にしない。

両GPUで実6形状とK2016/2048/2080/17376、N1023/1024/1025の境界を測定し、
72組のM=1比較を独立FP32 sampled oracleで確認した。全出力の非finite検査も実施した。
M>1のhost selector境界は1/2/16/17/127/128/129を検査し、gfx1201のM128 MXFP8 ID37と
M17 MXFP6 ID48はA16指定下でも従来のprovider・出力を維持した。

演算子wallのA16短縮倍率は、実6形状でV620 MXFP8 1.365〜1.494倍、MXFP6 1.251〜1.702倍、
R9700 MXFP8 1.158〜1.183倍、MXFP6 1.175〜1.303倍。baselineにはquantizer時間を含む。

## 実MTPの比較

8192入力／128出力、本体chunk2048、state8320、MTP幅2、MXFP8 E4 KV、
固定sampling seed123、stock auto、1 warmup＋3 measured。
V620はgfx1030／GPU-76a08c022586fed6、R9700はgfx1201／GPU-a8e9ddefa2d60f55。
全系列でHIP-only、非zero dispatch、fallbackなし、cleanup0、反復token hash一致を確認した。

既存benchmarkのMTP priming上限は1024だった。本体chunk2048とは別である。
計画のM=2048を満たすためbenchmark専用のpriming容量指定を追加し、両条件を保持する。
通常runtimeのpriming既定は変更していない。容量変更で生成履歴とblock数も変わるため、
BF16基準を異なるcampaignから混ぜない。

各時間・block数は3測定の中央値。decode tok/sはbenchmarkの127 decode tokenの時間率を使う
（最初の1 tokenはprefill出力）。非draftには検証に加えてsampling等の残り時間を含む。
各runの厳密な分解はdecode時間＝block数×(draft/block＋非draft/block)。
中央値同士の積・和は、同じrunの値とは限らない。

### MTP priming上限1024

| GPU | MTP行列 | decode tok/s | BF16比 | block数 | 採用/提案 | draft ms/block | 非draft ms/block | priming ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V620 | BF16 | 25.270 | +0.00% | 53 | 74/106 | 12.520 | 82.303 | 1510.077 |
| V620 | W8A8 | 21.454 | -15.10% | 58 | 70/115 | 18.493 | 83.524 | 321.904 |
| V620 | W8A16 | 23.718 | -6.14% | 54 | 74/108 | 16.124 | 83.035 | 320.143 |
| V620 | W6A6 | 22.842 | -9.61% | 56 | 71/111 | 14.954 | 84.332 | 331.216 |
| V620 | W6A16 | 22.289 | -11.80% | 60 | 68/120 | 13.091 | 81.877 | 328.084 |
| R9700 | BF16 | 34.596 | +0.00% | 50 | 78/100 | 10.412 | 63.007 | 112.494 |
| R9700 | W8A8 | 32.967 | -4.71% | 51 | 77/102 | 12.866 | 62.672 | 138.230 |
| R9700 | W8A16 | 28.235 | -18.39% | 60 | 68/120 | 11.994 | 62.783 | 137.276 |
| R9700 | W6A6 | 30.760 | -11.09% | 56 | 71/112 | 11.457 | 62.272 | 148.327 |
| R9700 | W6A16 | 32.336 | -6.53% | 54 | 74/108 | 10.588 | 62.144 | 147.882 |

### MTP priming上限2048

| GPU | MTP行列 | decode tok/s | BF16比 | block数 | 採用/提案 | draft ms/block | 非draft ms/block | priming ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V620 | BF16 | 25.049 | +0.00% | 53 | 74/106 | 12.847 | 82.808 | 1528.073 |
| V620 | W8A8 | 21.384 | -14.63% | 58 | 70/115 | 18.697 | 83.711 | 323.517 |
| V620 | W8A16 | 23.926 | -4.48% | 54 | 74/108 | 15.995 | 82.301 | 314.242 |
| V620 | W6A6 | 23.132 | -7.65% | 56 | 71/111 | 14.749 | 83.291 | 317.602 |
| V620 | W6A16 | 22.275 | -11.07% | 60 | 68/120 | 13.210 | 81.730 | 315.011 |
| R9700 | BF16 | 31.238 | +0.00% | 55 | 72/109 | 10.153 | 63.764 | 111.193 |
| R9700 | W8A8 | 32.286 | +3.35% | 51 | 77/102 | 13.389 | 63.729 | 141.972 |
| R9700 | W8A16 | 28.456 | -8.91% | 60 | 68/120 | 12.077 | 62.308 | 140.054 |
| R9700 | W6A6 | 30.325 | -2.92% | 56 | 71/112 | 11.797 | 62.941 | 152.197 |
| R9700 | W6A16 | 31.826 | +1.88% | 54 | 74/108 | 10.906 | 62.992 | 153.143 |

通常priming1024では全A16系列がBF16を下回った。計画指定priming2048では
R9700 W6A16が54 block対BF16 55 block、decode +1.88%となり、Stage 3の条件が成立した。
R9700 W8A8もこの条件ではBF16を上回るが、A16追加改善の対象形式はMXFP6に限定する。
V620 MXFP6やR9700 MXFP8ではA16でblock数が増え、演算子の短縮を実MTPの高速化に結び付けられなかった。
単一promptの生成履歴が分岐した結果であり、全タスクでの速度優位を示すものではない。

## Kernel activityの内訳

M=1、K5120、N17408、3 warmup＋32 measuredのrocprofv3 kernel trace。
以下はprofile下のactivity timestampで、非profileのwall時間とは区別する。

| GPU | 形式 | A8/A6 quantizer µs | A8/A6 matmul µs | A16 matmul µs | matmul短縮 |
| --- | --- | ---: | ---: | ---: | ---: |
| V620 | mxfp8 | 2.34 | 702.73 | 487.81 | 30.58% |
| V620 | mxfp6 | 2.36 | 433.45 | 320.94 | 25.96% |
| R9700 | mxfp8 | 2.64 | 613.06 | 530.06 | 13.54% |
| R9700 | mxfp6 | 2.80 | 471.06 | 374.36 | 20.53% |

A16ではquantizer dispatchが0。短縮の大部分はmatmul activity側にあり、
quantizerの起動省略だけでは説明できない。帯域律速か演算律速かの確定にはこのtraceだけでは不足する。

## priming追試とStage 3の採否

- primingのABBA追試を完了。A6中央値151.175 ms、A16 152.058 ms（+0.584%）。
  範囲は重なり、paired差の符号は混在し、この初期追試では非増加を実証できなかった。
  後述の区間別確認で非退行確認を完了した。
  `prime_mtp_prefix`の計測には先頭tokenのM=1 state-only処理も含まれるため、
  prefix全体をM=2048だけの非対象領域と扱わない。M>1のprovider維持は別の演算子検査で確認した。

Stage 3ではllama.cpp commit `bc52a12b38941b0a690ade65fbc5749715224e30`の
`mmvq.cu`／`mmvq.cuh`／`vecdotq.cuh`を先に参照した。MXFP6 E3M2 block32とBF16 activationに
そのまま対応するlayout／演算契約ではないため、sLLM既存codecとwave reductionで実験した。
外部sourceのcopy／adapt／portは行っていない。

R9700 MXFP6だけで、同一binary・3 warmup＋32 measured・全18形状を比較した。
各候補は実6形状＋境界のsampled FP32 oracle、全出力finite、単一dispatch、解放の検査を通過した。

| 候補 | 実験ID | 内容 | 実6形状の速度比（既存A16=1） | 採否 |
| --- | ---: | --- | ---: | --- |
| scale broadcast | 103 | waveでscale codeを共有するcontrol | 0.774〜0.940 | 不採用 |
| packed＋pair scale-once | 104 | block24 byteを6 dwordで読み、pair和の後にscale適用 | 0.472〜0.736 | 不採用 |
| packed＋指数加算 | 105 | 同じpacked load、normal範囲の重み復号を指数加算 | 0.753〜0.943 | 不採用 |

3候補とも18形状すべてで既存ID102を下回った。Stage 3による高速化は確認できず、
候補を通常sourceから撤去してStage 1のA16を保持する。Stage 3候補の実MTP高速化は主張しない。
全形状で遅い候補の追加full-model計測と追加extreme fixture検査は行わず、未検証範囲として残す。
最終的に保持するStage 1経路の実MTP結果は上記Stage 2の両GPU・全5系列である。

初回HIP buildはshuffle lane引数のsignedness警告をerrorとして検出した。
範囲が0〜31の引数へ明示int castを加えたr2 buildは両targetで成功し、このbinaryで測定した。
prototype source・binary・raw計測はignored artifactへ保存し、Gitへ追加しない。

撤去後のfresh両target buildとhost検査をPASS。最終GPU72ケースは全出力・重みdigestと
kernel ID／symbolがStage 1と一致した。最終full-modelの速度表はStage 2計測であり、
撤去後に同じfull-model campaignを再実行した値ではない。追加GPU検査は演算子に限定した。
サービスのunit／binary／run.sh hash維持、health／ready HTTP200と両GPU auto復元を確認した。


## 区間別primingの最終確認

先頭M=1とbatchを別々に計測する診断を明示opt-inで追加した。
R9700 MXFP6、同じ8192/128・priming2048・ABBA4job・各1 warmup＋3 measured。
生成token列は両形式とも前のStage 2と一致し、処理区間はすべて
1／2048／2048／2048／2047行だった。各callはKV書込み完了まで含むhost時間であり、
非同期投入だけの時間やGPU kernel timestampの総和ではない。

| 区間 | A6中央値 ms | A16中央値 ms | 相対時間差 |
| --- | ---: | ---: | ---: |
| 先頭M=1 | 1.784575 | 1.721877 | -3.5133% |
| M=2048の3回合計 | 115.505829 | 115.642368 | +0.1182% |
| M=2047末尾 | 35.208585 | 35.216435 | +0.0223% |
| 後続batch4回合計 | 150.840440 | 150.841987 | +0.0010% |
| 総priming | 152.710951 | 152.589878 | -0.0793% |

各欄はrunごとの値から求めた中央値なので、欄同士の和は合計欄と一致するとは限らない。
以前の+0.584%は再現せず、総primingの非増加と後続batchのほぼ同等な時間を観測した。
これで非退行確認を完了するが、0.08%程度の差を有意な速度向上や全条件の保証へ一般化しない。

診断buildはhostテスト10件とgfx1201 buildを通過した。共通HIP sourceは撤去後72ケースを
検証したbuildとhash一致で、追加変更はRustのopt-in時間記録とbenchmark reportのみ。
全4jobでHIP-only、cleanup0、従来生成hash一致を確認し、サービスhashとhealth／ready、autoを復元した。

[実行計画](../../../../plans/archive/2026/09/11-20/phase85-m1-a16-mtp.md) /
[集約結果](../../../../../ci/matrix/phase85-a16-mtp-results-v1.json)
