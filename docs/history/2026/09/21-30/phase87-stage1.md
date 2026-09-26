# Phase 87 段階1: NVFP4 W4A4 decode

2026-09-25着手。段階2を完了した後、両GPUの実モデルM=1をID84で最適化する。
本番shapeはwide `(1,5120,17408)` 112回/token、down `(1,17408,5120)` 56回/token。
M=2〜3のMTP verifyはID94を対照とし、変更しない場合も選択を確認する。

WU0のwarm read参照に対する両shape合計の正の余地はV620 0.8320、R9700 1.4121 ms/token、
探索継続線は0.4160／0.7061 ms/token。これらはread-only観測を使った探索値であり、
物理DRAM上限や採否の自動条件ではない。WU-D1は、直前のGQA型KV readerだけで
NVFP4 M=1 kernelが約9〜10%遅くなることを示した。このため単体の孤立値だけで
候補を採用せず、同じ先行処理の後でcontrol／候補を交互に測る。

現行ID84はFP32 scale LUTのbank padding、non-temporal load、gfx1030の2-block／
gfx1201の4-block weight先読みを実装済み。C1はwave共通weight／scale列baseの明示scalar化だけを試す。
E2M1 byte permute、DP4A／sudot4、scale、FP32 FMA順、BF16 RNEは維持する。
新しい外部codeはコピー・移植しない。

## 実施結果

### C1: wave共通weight／scale基点の明示scalar化

既存ID84のE2M1 byte permute、DP4A／sudot4、LUT、FP32 FMA／reduction、BF16 RNEを
共有し、weightとblock scaleのwave共通列baseだけを`readfirstlane`でscalar化した。
最初のstandalone probeはhost launcherがdevice専用macro `__gfx1030__`でdynamic LDS量を
決めていたため、host compile側で0 Bを渡し、候補がゼロ出力となった。この失敗runは
数値・性能証拠へ使わない。`K/16*20` Bをhost側で渡すよう修正し、さらに候補を本番ID84と
同じlowp翻訳単位のtest-only symbolへ置いて別archiveを生成した。本番selectorは未変更。

両GPUの両tupleで全出力がproduction ID84とbitwise一致、独立FP32 oracle最大0 ULP、
finite／repeat／guard／cleanup PASS。canonical UUID、Qwen service停止、R9700 service lease、
performance level復元、source／archive／binary hash前後一致をrunnerで確認した。
各shapeは512 MiB以上の巡回weight pool、300 ms継続warmup後、同一processの
AB-BA-AB各9 samplesでisolatedとGQA型KV reader先行の両条件を測った。

| GPU | M1 shape | condition | control dot ms/call | C1 dot ms/call | 形状の回数を掛けた短縮 ms/token | 全3roundの方向 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| V620 | wide K5120,N17408（112回） | isolated | 0.121841 | 0.120121 | +0.192640 | 改善 |
| V620 | wide | GQA先行 | 0.135441 | 0.131761 | +0.412160 | 改善 |
| V620 | down K17408,N5120（56回） | isolated | 0.119681 | 0.120401 | −0.040320 | 退行 |
| V620 | down | GQA先行 | 0.136640 | 0.126961 | +0.542024 | 改善 |
| R9700 | wide | isolated | 0.102281 | 0.104841 | −0.286720 | 退行 |
| R9700 | wide | GQA先行 | 0.103000 | 0.105160 | −0.241920 | 退行 |
| R9700 | down | isolated | 0.103041 | 0.100360 | +0.150136 | 改善 |
| R9700 | down | GQA先行 | 0.103801 | 0.101241 | +0.143360 | 改善 |

V620のGQA先行2形状を合計すると約**0.954184 ms/token**のdot短縮で、
探索継続線0.4160 ms/tokenを上回る。各shapeとも全3roundの方向が正である。
一方、R9700のGQA先行合計は**−0.098560 ms/token**で、wideの退行がdownの短縮を上回る。
R9700にC1を既定採用せず、V620だけをproduction経路へ接続して通常8192/128の
MTPなし／ありで効果と他経路の副作用を検証する。孤立条件のV620 downは微小退行しており、
GQA先行条件の利益を通常モデルへ自動換算しない。

V620 rawは`.local-artifacts/phase87/stage1/c1-gfx1030-both-gqa-r2/`、
`execution.json` SHA-256は`cac9261383f5b2af5ea9dc09cab428950e84ead5782e9d872f6b0994ce0dd1a2`、
binary SHA-256は`ae260a076be03f7d46c35d95744088c7511c665c1d2a51ad16d33fc319270a34`。
R9700 rawは`.local-artifacts/phase87/stage1/c1-gfx1201-both-gqa-r1/`、
`execution.json` SHA-256は`fc9aaff47615e02e2ff68226f42cf8d3dc8cacb9e9a1c7a110735fa9c04f469b`、
binary SHA-256は`e7789e056e31617d01c856516d6f3ff21316ebde6c976b74d289a3086e59f6ae`。

### 次の検証

V620 C1をexact target／M=1の2 tupleだけに接続し、M=2〜3のID94を維持したまま
公開dispatch、数値、通常8192/128のMTPなし／ありを確認する。通常TPOTと副作用が採否条件に
届かなければC1を本番から外す。R9700候補はtest-onlyに留める。

### V620公開経路の暫定統合結果

V620だけでexact M=1 wide/downをSGPR kernelへ向け、M=2〜3と他shape、R9700の既定は
そのままにした。公開launcherから呼ぶkernelを`rocprofv3`で監査し、候補symbolの起動2回と
候補直接呼出しとの全出力bitwise一致を確認した。trace SHA-256は
`eacb07e9b839616be31d14d67ff9e0d8c989339e40783df4102f02d0603331bf`。

公開HIP runtimeのV620 archiveを直接リンクした専用probeでも、両tupleの全出力が
旧ID84とbitwise一致・独立oracle最大0 ULP。GQA先行の3roundは両shapeで短縮方向だった
（`.local-artifacts/phase87/stage1/c1-gfx1030-current-public-r1/`、report SHA-256
`f757dfc00cef4a9563899877c73fbe9bacc2633b7925d2eafb4cd099223c92ed`）。
同archiveのresource queryは旧ID84→候補でVGPR 63→53、static LDS 1,056 B、
dynamic LDSはK5120で6,400 B／K17408で21,760 B、256 threadのactive blocksは8／2のまま。
resource原票SHA-256は
`dd55733289390adcd9a649e480b2ef1594845f6979a895a44e1a7b0d185c2723`。

通常8192入力／128出力、MXFP8 E4 KV、固定GPU sampling、各1 warmup＋3 measuredの
V620比較は次のとおり。MTPありは較正済みNVFP4 companionと縮小draft headを使用した。

| 経路 | control TPOT ms | C1 TPOT ms | C1短縮率 | 生成SHA・peak VRAM |
| --- | ---: | ---: | ---: | --- |
| MTPなし | 58.068919 | 57.385953 | **+1.176%** | 同一、23,914,018,344 B |
| MTPあり | 30.416029 | 30.405897 | +0.033% | 同一、25,729,515,552 B |

MTPなしの3 measuredすべてで候補TPOTはcontrol 3 measuredの最小値より短かった。
prefill中央値はMTPなし35.826→35.785秒、MTPあり36.198→36.289秒。
MTPありの差0.091秒は両群のmeasured範囲が重なり、今回の速度差へ帰属しない。
MTPありの受理／提案は前後とも75/103。全runがHIP-only、fallbackなし、128 token、finite、
request/session cleanup 0でPASSし、GPU性能levelは`auto`へ復元された。

対照binaryはWU-3Pの`c0cadfaa5b2e7ccf4016bf307511bb06ed84a50c4099a40500aa524417a8d373`、
候補binaryは最初のsymbol名を使った
`17b5fe9e58724ac6f0e719cfacb2191301601c38108c7beefbd8db18f91cd4af`。
rawは`.local-artifacts/phase87/stage1/baseline-full-gfx1030-v1/`と
`candidate-full-gfx1030-v1/`、execution report SHA-256は順に
`2d759569192f9fb199203f5ed85e3b411792687da419d5c0a98243dbe74e2a32`／
`2038c177b655718875a597fdbb06739d85e2f410283f0bedcbed585678e91d03`。

この版では重い変更の全体1%条件を満たしたが、symbol改名・metadata同期後の再buildで
短縮が0.969%となった。採否を確定せず、同じC1仮説のwave共通base計算をKループ外へ移した。
上表はその前の測定値であり、最終採否の値には使わない。

### C1 hoist最終結果とV620限定採用

当初のC1はprefetchのたびに同じwave共通列baseへ`readfirstlane`を繰り返していた。
V620のID73 body内でweight／block-scale baseをKループ前に一度だけscalar化し、
2-block先読みloaderへ渡した。E2M1 byte permute、4回DP4A、LUT、FP32 FMA／reduction、
BF16 RNEは共通helperのまま。gfx1201のID67 bodyとM=2〜3のID94は変更していない。
logical providerはID84を維持し、exact gfx1030 M=1 wide/downだけ
`sllm_nvfp4_w4a4_decode_scale_lut_gfx1030_sgpr_v1`へ向ける。runtime切替は追加しない。

現在sourceのV620両shapeは旧ID84と全出力bitwise一致、独立FP32 oracle最大0 ULP、
finite／repeat／guard／cleanup 0。GQA reader先行、300 ms継続warmup、512 MiB超weight poolの
同一process AB-BA-AB各9 samplesでは、wideが0.135964→0.129684 ms/call、
downが0.135884→0.125644 ms/call。112／56回を掛けると**1.2768 ms/token**のdot短縮で、
両shape・全3roundが改善方向。孤立条件のwideは短縮したがround間に大きな外れ値があり、
採否には実モデルのGQA先行条件と通常モデルの値を用いた。
rawは`.local-artifacts/phase87/stage1/c1-hoist-gfx1030-both-v1/`、execution SHA-256は
`8762a98e043c24c258ceec999745f81088620cadf913deb4fd9f2d68dde9d372`。
最終archiveのresource queryでは旧ID84→新kernelがVGPR 63→49、static LDS 1,056 B、
dynamic LDSはK5120で6,400 B／K17408で21,760 B、256 thread active blocksは8／2のまま。
resource原票SHA-256は
`884dd9b1f27f32ead8e3794248a4f94ca85babeccbe802e09dcb9124a6ef1647`。

保存済みWU-3P control binaryと最終hoist binaryで、V620の通常8192入力／128出力を
各1 warmup＋3 measuredにした結果は次のとおり。両経路の生成SHAはcontrolと一致し、
MTPありの受理／提案も75/103で一致した。

| 経路 | control TPOT ms | 最終hoist TPOT ms | 短縮率 | peak VRAM |
| --- | ---: | ---: | ---: | ---: |
| MTPなし | 58.068919 | 56.990410 | **+1.857%** | 前後とも23,914,018,344 B |
| MTPあり | 30.416029 | 30.414372 | +0.005% | 前後とも25,729,515,552 B |

MTPなしの候補3 measuredは56.975／56.990／57.050 msで、controlの最速58.061 msより
すべて短い。MTPありのTPOT中央値は差0.002 ms未満。prefill中央値はMTPなし
35.826→35.813秒、MTPあり36.198→36.322秒。MTPありの差0.124秒は両群の
measured範囲が重なり、同じprefill経路の測定揺れを超える退行とは判断しない。
全runはexact UUID、HIP-only、fallbackなし、128 token、finite、cleanup 0でPASSし、
performance levelを`auto`へ復元した。

最終binaryは追跡対象外の`.local-artifacts/phase87/stage1/bin/bench-gfx1030-hoist-v1`
（SHA-256 `1c78b2f7b2b2ae229b12851ad4df646901939175738818e5ebb768bbea4559a3`）。
MTPありcontrol rawは`.local-artifacts/phase87/wu3p/candidate-gfx1030-v1/nvfp4-baseline/report.json`
（report SHA-256 `00c9429efe738e903a7e55d5f7b0b4d2f3167b0e7f5862c0842405e2f1271978`）、
その`execution.json` SHA-256は
`0968475892180c65097e5fc7af8d4f4d110f3fa1b76c36ac47dc8108f14fb803`。
MTPなし原票`.local-artifacts/phase87/stage1/hoist-off-gfx1030-v1/`のexecution SHA-256は
`3f9eeb30f572a9e2d9220c6d42ad9a192444fd5e3d7a65fbfb7504ac69756a40`、
MTPあり原票`.local-artifacts/phase87/stage1/hoist-on-gfx1030-v1/`は
`c388a4e97653143aeb65445bf51e7738cbda0c02c7532ed5a61616f06aa2a4c2`。
最終公開launcherのtraceで新symbolを2起動確認し、直接呼出しとbitwise一致した
（trace SHA-256 `650c3eb00c01b143d4848c95963f64082ebfc61e74f3a947123c73bff068f949`）。
最終sourceを`clang-format`した後のgfx1030 release rebuildは、上記の計測binaryと
full SHA-256が一致した。gfx1201のlowp compile-only、affected public runtime host fixture 1/1、
H3／public-runtime H3／RMSNorm H3の契約validator、Markdown local link、`git diff --check`もPASS。
dirty checkoutに対するstrict H3 artifact runnerは正式なimmutable evidenceへ使わず、
draftのcompileとhost契約をそれぞれ区別した。統合レビュー1回の指摘はMTPありcontrol report SHAの
転記漏れだけで、原票照合後に修正・焦点再レビュー済み。新しい外部project codeのimportはない。

共通ルールの変更部分全round改善、通常TPOT 1%以上、N0、MTPあり・prefill・VRAMの
副作用非退行を満たすため、**V620のexact M=1 2 tupleへC1を採用する**。
R9700は単体でwide退行が大きかったため従来ID84を維持する。
段階1全体ではR9700の残余とM=2〜3の評価を続ける。

### R9700 hoist追試とC1の打ち切り

V620採用版のKループ外base hoistをR9700 `gfx1201`のID67本体にもtest-onlyで適用し、
実M=1 wide `(1,5120,17408)`を現行ID84と同一processで比較した。全出力bitwise一致、
独立FP32 oracle最大0 ULP、finite／repeat／guard／cleanup 0。exact UUID
`GPU-a8e9ddefa2d60f55`、512 MiB超のweight pool、300 ms warmup、AB-BA-AB各9 samples、
isolatedとGQA32先行の双方で実行した。R9700 serviceはinactive、performance levelは
`auto`へ復元した。

| 条件 | ID84 dot ms/call | hoist dot ms/call | ID84 total ms/call | hoist total ms/call |
| --- | ---: | ---: | ---: | ---: |
| isolated | 0.102681 | 0.104881 | 0.117161 | 0.119601 |
| GQA32先行 | 0.103121 | 0.104801 | 0.237642 | 0.239202 |

dotは双方の条件で全3roundが退行方向だった。GQA32先行のwide 112回だけで約
0.188 ms/tokenの増加となる。以前のnon-hoist C1で得たdown 56回の短縮
0.143 ms/tokenを仮に維持しても合計は正にならず、R9700の探索継続線
0.7061 ms/tokenからも遠い。このためdownのhoist再計測と通常モデル再実行へ進まず、
R9700候補symbolとtest-only codeを撤去した。R9700 productionはID84のままとする。

数値と性能の原票は`.local-artifacts/phase87/stage1/phase87-stage1-r9700-sgpr-wide-r1/`と
`phase87-stage1-r9700-sgpr-wide-perf-r1/`へ保存した。後者の`execution.json` SHA-256は
`6af4cae77c1fb543b390969ff08bf8714c2a8d011a4af59cc65e506e721459ea`、
計測binary SHA-256は`3d22e72317cdec22f9075ea54fc092261886fad776e0e7c25773eae1889d5586`。

### M=2〜3のID94対照と段階1完了

現行M=2〜3は両GPUともexact Qwen wide／down tupleでID94を使う。既存の4-row
activation共有LDS案はPhase83.5で大幅退行しており、別構造の候補として
`TM=ceil(M/16)=1`、split-K=2をtest-onlyで試した。Kを2つの連続範囲へ分け、
ID94の行間weight再利用を維持し、専用reduction kernelで合成するN1構造である。
まずV620 `gfx1030`のM=2 wide `(2,5120,17408)`を測った。

全出力はID94とbitwise一致、独立oracle最大0 ULP、finite／repeat／guard／cleanup 0。
producerはVGPR 75、static LDS 1,056 B、256 threadでactive blocks 6、
reducerはVGPR 6、active blocks 8。512 MiB超のweight pool、300 ms warmup、
同一process AB-BA-AB各9 samples、isolatedとGQA32先行の両条件で実行した。

| 条件 | ID94 dot ms/call | split-K＋reduction ms/call | ID94 total ms/call | candidate total ms/call |
| --- | ---: | ---: | ---: | ---: |
| isolated | 0.124120 | 0.159920 | 0.138201 | 0.173920 |
| GQA32先行 | 0.125360 | 0.160361 | 0.301961 | 0.333401 |

dotは全3roundで約0.035 ms/call遅く、GQA32先行では約27.9%増だった。
reduction追加とproducerの負荷がweight再利用の利点を上回り、read参照余地の半分にも
向かわなかったため、M=3とR9700への拡張は打ち切った。test-only code以外のselectorは
変更せず、**両GPUのM=2〜3はID94を維持する**。実モデルのMTPあり再実行は、
この不採用候補に対しては行わない。

rawは`.local-artifacts/phase87/stage1/small-m-splitk-gfx1030-m2-v2/`、
`execution.json` SHA-256は
`abc3354542c8e46fed1e8c586d9bc71f2264a98345b3c837571f01b4c66751f6`、
binary SHA-256は`59e001f3b46063c7f82bb8373564bf1e6a360a8a2817221017d7e6b8ecb61213`。
exact UUIDは`GPU-76a08c022586fed6`。Qwen service停止、performance level
`auto`復元と入力source hash不変をrunnerで確認した。外部project codeのimportはない。

V620 M=1の2 tupleだけをC1へ採用し、R9700 M=1と両GPU M=2〜3の既定は維持して、
**段階1を完了**とする。

対応する計画: [Phase 87 段階1](../../../../plans/archive/2026/09/11-20/phase87-qwen38-nvfp4-single-request.md#段階1-nvfp4-w4a4のm1-decode)。
