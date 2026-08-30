# Phase 54: KV FP8 block16精度改善研究

## 状態

2026-08-27に開始し、同日`no-improvement`で完了した。primary targetはexact gfx1030／V620／E5M2とした。
finalistがなかったためtransfer targetのexact gfx1201／R9700／E4M3とMI300Xは実行せず、後続の独立format研究まで延期する。

## 固定baseline

- production E5 block16 v2: KLD p99 `0.03659844555378746`、top-1 `0.9`、long-context loss
  `0.08333333333333337`。
- standard MXFP8 E5 block32: KLD p99 `0.03218873133110086`。
- local-MSE／parent32-guardは`0.04063529273873547`で棄却済み。
- normalized H16はproduction controlと同じ`0.03659844555378746`で棄却済み。direct GPU oracleではK byte／scaleの変化、
  V不変、Q/K同時変換後のattention数値一致を確認しており、未適用ではない。

## 開始時の実装判断

Phase 53 quality schema／aggregatorはdescriptor v2、`StandardMxFloorPowerV1`、3 repeatへ固定されているため、研究candidateを
Phase 53 reportとして発行しない。Phase 54専用schema／runnerへcandidate specとそのdigest、actual binary、1／3 repeat、per-case
logit差、完全直列cleanupを結合する。

現行KV stateはK/Vで一つのencodingを共有するため、K-only／V-onlyのためにproduction mixed-plane ABIを追加しない。最初の
attributionはresearch feature限定のFP16-state block16 roundtrip surrogateで行い、actual block16 K+Vと一致する範囲を確認する。
wave 1のscale候補は既存production Floorを変更せず、research buildだけに別append kernelを追加してK/V別Ceil／NearestEvenExponentを
同一binaryから選ぶ。MXFP8 comparatorは変更しない。

## 2026-08-27: research harnessとwave 1 strict selector

production descriptorを偽装しないPhase 54専用quality runnerを追加した。candidate specはK/V別の閉じたrecipe、rounding、transform、
calibration、descriptor compatibilityとcanonical digestを持つ。1または3 repeatだけを受理し、各repeatをFP16→production block16→
candidate block16→MXFP8の順に完全解放してから実行する。per-caseのprefill/decode KLD、top-1、NLL、最大logit差と最初の分岐を保存し、
HIP-only、fallback 0、terminal-zero cleanupをfail-closedにした。

研究runtimeは通常buildではFloor/Floor以外を拒否し、`SLLM_ENABLE_PHASE54_KV_RESEARCH=1`の別buildだけK/V別
Floor／Ceil／NearestEvenExponentを選択する。Floor/Floorは既存production kernelへ直接dispatchし、MXFP8経路は変更しない。
host oracleも同じscale規則を独立実装し、Floorとproductionのbyte完全一致、15／16／17、255／256／257、NaN／Inf／subnormal／
signed zero、E8M0境界を確認した。GPU側で発見したsigned-zero-only blockの非canonical `-0` payloadはunit scale／positive-zeroへ修正した。

exact gfx1030／GPU UUID `GPU-76a08c022586fed6`、同一research binary SHA-256
`7beb9d1a7cc49eea95abbba77a3a2bafaf4413101723b8a78749b819680767b0`、Qwen3.5-4B model fingerprint
`sha256:f143d7b504170d071c77818105f7a07dc0297c6bea0c61a5404b071fed0c1fae`、dataset SHA-256
`a2252d882ffd7e1fbb546d86b2b573bd2410467382c7da874f4fbd3dc8adc77d`で探索1回を取得した。

| candidate（K／V） | KLD p99 | top-1 | task delta | long-context delta | 採否 |
| --- | ---: | ---: | ---: | ---: | --- |
| Floor／Floor control | `0.03659844555378746` | `0.9` | `0` | `0.08333333333333337` | production完全再現 |
| Ceil／Floor | `0.03659844555378746` | `0.85` | `0` | `0.08333333333333337` | top-1悪化で棄却 |
| NearestEven／Floor | `0.03659844555378746` | `0.85` | `0` | `0.08333333333333337` | top-1悪化で棄却 |
| Floor／Ceil | `0.04331390780013198` | `0.9` | `0` | `0.08333333333333337` | KLD悪化で棄却 |
| Floor／NearestEven | `0.04331390780013198` | `0.9` | `0` | `0.08333333333333337` | KLD悪化で棄却 |
| same-run MXFP8 comparator | `0.03218873133110086` | `0.8` | `0` | `0.16666666666666663` | 明示比較だけ |

Floor/Floor candidateとproduction controlの全logitは完全一致した。CeilとNearestEvenはこのdatasetではK-only同士、V-only同士の集約値も
一致し、上向きscale選択はK側でtop-1、V側でKLDを悪化させた。したがって組合せ候補へ進まず、strict selector wave 1を棄却した。
quality report digestは順にFloor/Floor `3e0f92e0...d9e9`、Ceil/Floor `f506736f...943e`、
NearestEven/Floor `bed24c65...bc2b`、Floor/Ceil `f332a02d...d6f6`、Floor/NearestEven `96cd4c74...c4c5`である。

各candidateには別のPhase 54 direct GPU oracleも実行した。6境界caseにsigned-zero-onlyとrecipe識別caseを加えた8 append、K/V value／
scale byte exact、tail zero、KV長2・非zero queryのK-sensitive scalar attention oracle、HIP-only、fallback 0、cleanup 0を5候補すべて
PASSした。非Floor reportはproduction descriptor v2を持たない。raw reportはrepository外
`external:phase54/gfx1030/direct-*.json`へ保存し、SHA-256はFloor/Floor `b82f96cb...99e9`、Ceil/Floor
`4a3e429e...5cc`、NearestEven/Floor `f3516860...9fd`、Floor/Ceil `4bbff79d...f31`、Floor/NearestEven
`03b94231...86a`である。各実行後にGPU0 VRAM、retryable cleanup、durable quarantineが通常値／0／0へ戻った。

## attribution継続

production mixed-plane ABIは追加せず、research feature限定でFP16 stateの最初のfull-attention layer 3へK-only／V-only／K+Vの
Floor block16 quantize-dequantize roundtripを注入する。通常buildにはselector、readback、traceを含めず、active時はstate image、prefix、
checkpointのimport／export／forkを拒否する。host componentとfeature／normal buildはPASSした。

同じbinary SHA-256 `30038688ead3bce7ef211301f6646959729be7b64202145a1fbed63642a05156`で、reviewed full-attention
8層それぞれのK-only／V-only／K+Vを一回ずつ、合計24 run測定した。各runはFP16 Off residentを解放してからFP16-state roundtrip
residentを作り、HIP-only、fallback 0、audit semantics／layer一致、terminal-zero cleanupを確認した。

| layer | K-only KLD / top-1 | V-only KLD / top-1 | K+V KLD / top-1 |
| ---: | ---: | ---: | ---: |
| 3 | `0.0007571` / `1.0` | `0.0009749` / `0.85` | `0.0013840` / `0.8` |
| 7 | `0.0007430` / `0.95` | `0.0019468` / `0.9` | `0.0019468` / `0.95` |
| 11 | `0.0003981` / `0.95` | `0.0009792` / `0.95` | `0.0009792` / `0.95` |
| 15 | `0.0006772` / `1.0` | `0.0013903` / `0.95` | `0.0013903` / `0.95` |
| 19 | `0.0009645` / `1.0` | `0.0398337` / `0.9` | `0.0398337` / `0.95` |
| 23 | `0.0005930` / `1.0` | `0.0007354` / `1.0` | `0.0006319` / `0.95` |
| 27 | `0.0005693` / `0.95` | `0.0006011` / `0.9` | `0.0010033` / `0.95` |
| 31 | `0.0010435` / `0.95` | `0.0023586` / `0.8` | `0.0023586` / `0.85` |

K-onlyは全層でKLD p99 `0.0003981–0.0010435`に留まった。一方V-onlyはlayer 19が`0.0398337`と突出し、layer 31も
top-1 `0.8`／long-context loss `0.16666666666666663`だった。従ってこのsurrogateではV、特にlayer 19と31が主要因である。
raw reportはrepository外`external:phase54/gfx1030/attribution-matrix-v1/`に保存した。これはFP16-state単層roundtripの因果診断であり、
全層production block16の数値を再現する証拠ではない。

wave 2の最小controlとして、Q/Kへ同一の16×16転置permutationを全full-attention層で適用するcandidateを選んだ。mappingは
`out[i]=in[16*(i%16)+floor(i/16)]`、self-inverseで量子化前QKを保存し、Kのblock16 groupingだけを変更する。V/Oは不変、scaleは
production Floorのまま、transform identity digestは`806cc66a1135d36fe594c96c78b1329efb955f94a30e9664c20e3d0e41c0cef6`である。
K-onlyが主要因でないため勝者期待は限定的だが、V/O foldを伴う広い候補へ進む前の低リスクcontrolとして評価した。

candidate `phase54-kq-transpose16x16-all-full-v1`はexact gfx1030 direct GPU oracleをPASSした。research direct binary SHA-256は
`8e09d96b943644efbcc1fb0aa018b153bf66204f616e89dc25fecf8acec81ee5`で、GPU格納K value／scale byte exact、V不変、
FP64 QK最大差`0`、KV長2 attention数値一致、HIP-only、fallback 0、cleanup 0を確認した。品質binary SHA-256は
`1ec38ee86b13f4ca6cd8fea9a5bf0f7ac4ee11a8a56a1bc57844b6919f1f8336`、report digestは
`479fd11815c26f398ea31c7590e1e14e9758fba90381aba8f170e4de5e8c1f11`である。

一回品質はproductionと同じKLD p99 `0.03659844555378746`に留まり、top-1は`0.9`から`0.8`、long-context lossは
`0.08333333333333337`から`0.16666666666666663`へ悪化した。MXFP8のKLD p99 `0.03218873133110086`にも届かないため棄却した。
これはK groupingだけでは主因を解消できないというattribution結果と整合する。次候補はV-only寄与が突出したlayer 19に限定し、Vへ
同じpermutationを適用してattention outputでself-inverseを掛け、量子化前V/O意味を保存する。

layer 19 V/O candidate `phase54-vo-transpose16x16-layer19-v1`は、transform digest
`7da862b274ac32124e4a7b2550ed947fe865140ddaa2fd940a89ebfa9d8c4ad4`を持つ。Vは`[tokens,4,256]`の各KV head、
attention outputは`[tokens,16,256]`の各query headへ同じself-inverse permutationを適用し、後者をattention直後・SigmoidMul前へ置いた。
direct binary SHA-256 `500483e256eea5029ff352556fc241053116736aeeb3e842fdd5b3edd745cb6f`で、変換V byte／scale exact、
K/Q不変、V/O FP64最大差`0`、attention一致、HIP-only、fallback 0、cleanup 0をPASSした。

品質binary SHA-256 `74d6167ca1a9a27e7bcde9cd6b8023afdd68cd69752a69efa599b89c5f41982e`、report digest
`5aeec3188ddc0d8408a242046b74f0d6fda62b4591495998f697455ace9aa0a4`の一回品質では、KLD p99をproductionの
`0.03659844555378746`から`0.033918254226008415`へ改善し、top-1 `0.9`、task delta `0`、long-context loss
`0.08333333333333337`を維持した。ただしMXFP8 `0.03218873133110086`を下回らないためfinalist条件未達である。
次に追加寄与が見えたlayer 31も同じV/O変換へ加える`[19,31]`候補を一回だけ評価する。

`phase54-vo-transpose16x16-layers19-31-v1`はtransform digest
`5439e11e91b4c2acfd060fb1ec4d8f5fee2f1244e28c3e6588f2202fbe8e9a74`を持ち、layer 19と31だけへ同じV/O
companionを適用する。direct binary SHA-256 `01bbb14fd5eb5ccc1250fcff8978b196bfa1733121ee7223e05342ca5bbc9068`で、
semantic layers `[19,31]`、V/O FP64最大差`0`、GPU byte／scale exact、attention一致、HIP-only、fallback 0、cleanup 0をPASSした。

品質binary SHA-256 `57f1fcf4a2519cd4f93fef2479380a6140c15b5c807c04153b9ba641a0e2a029`、report digest
`140d83ffed02788face1a38e358d9e7cc810b3c33a98aad048dfbbaf557922a9`の一回品質はKLD p99
`0.03337377972334127`でlayer 19単独よりさらに改善したが、MXFP8 `0.03218873133110086`を下回らず、top-1も
production `0.9`から`0.8`へ悪化したため棄却した。task delta `0`、long-context loss `0.08333333333333337`は維持した。

## 最終判定

exact gfx1030／E5M2のPhase 54判定は`no-improvement`とする。scale recipe 4候補、Q/K meaning-preserving control、V/O
layer 19、V/O layers 19+31を同じmodel／dataset／metricと完全直列residentで比較したが、production block16 v2とstandard MXFP8
block32の両方をKLDで上回り、かつ他metricを悪化させない候補はなかった。最もKLDが低かったlayers 19+31はtop-1を悪化させ、
他metricを維持したlayer 19単独はMXFP8未達だった。

finalistがないためfresh 3-repeat、resource／性能、gfx1201 E4 transfer、MI300X実機は実行しない。省略時FP16、空default mapping、
production descriptor v2／`StandardMxFloorPowerV1`を維持する。Phase 54のselector、host readback変換、attribution、runnerは
`phase54-research` compile feature限定で通常buildから除外され、公開descriptor／state identity／defaultへ入らない。研究証拠の再現と
次の独立format研究に使えるようfeature sourceは保持するが、製品経路としては採用しない。

最終統合レビューでdirect evidence runnerのattention PASS述語が非direct dispatchを許し得る問題を検出した。attention caseには
direct dispatch、数値一致、K寄与の三条件をすべて必須化し、各条件を反転するmutation unit testを追加した。修正後はdirect runner
10 test、strict clippy、Phase 54 direct contract 7 test、format／diff checkをPASSした。取得済みGPU reportは修正前binaryのものだが、
各report自身が`attention_direct=true`、数値一致、K寄与を記録し、外部schema検証もPASSしているため、既存数値証拠は維持する。

## 2026-08-27: MXFP8 parent32 scale複製follow-up

ユーザー指示により、公開descriptor v2を変更せず、research recipe `parent32-duplicate`を追加した。各連続32値からstandard MXFP8と
同じE8M0 scaleを一度だけ決め、対応する2個のblock16 childへ同じscale byteを複製する。rounding、物理FP8 variant、K/V groupingは
MXFP8と同一に保ち、all-zero parentはscale `127`／正のzero payloadへcanonicalizeした。raw value bufferは非整列head dimensionの
row padding幅がblock16とMXFP8で異なるため同じ配列長にはならないが、有効な論理lane、復号値、scale対応は比較できる。

host oracleはOCP E4M3／E5M2、2 rows、head dimension `15,16,17,31,32,33,255,256,257`、片側16値だけzeroのparent、
signed zeroを含め、全論理FP8 value byte、`block16_scale[child] == mxfp8_scale[child/2]`、全復号値の完全一致をPASSした。
exact gfx1030／V620 E5とexact gfx1201／R9700 E4のdirect GPU oracleも8 append、K/V value／scale byte exact、KV長2 attention、
HIP-only、fallback 0、cleanup 0をPASSした。direct report SHA-256はgfx1030
`c8746b30561fd6e4392fc3e824a2bbbdf0b6b15ade4c140515456b16bc8e32fc`、gfx1201
`11ea481b757689483a49633210e5d1d1958f826934bd7be841f943c7e714e4d9`である。

Qwen3.5-4Bのfreeze済み20 measurementを両targetで完全直列実行し、candidateとsame-run MXFP8の全prefill／decode logitを
FP32 bit列で比較した。不一致時にreportを発行しないfail-closed checkを最終binaryへ追加した上で両runがPASSし、保存された
per-case／aggregateも完全一致した。quality report SHA-256はgfx1030
`ac247f80919ba65fb33740cbb1b115a6463cc142ec3af5d05b4d37da0010d8f0`、gfx1201
`fa9498cdafaaa73eab4ecb7b3aef7f7b108b120e95eb12676315bc0fa6822fe3`である。したがって同一OCP variantでは、parent32 scaleを
複製したblock16がMXFP8を表現・full-model logitとも完全再現することを確認した。これは実験結果であり、production v2、FP16 default、
空mappingの変更は行わない。FNUZ／gfx942はstandard MXFP8と同じ物理variantでないため本follow-upの一致主張に含めない。

## 2026-08-27: V620 E4M3／E5M2 attention速度follow-up

E5M2をV620で採用した当初の「FP16へ変換しやすく、E4M3より速い」という仮説を直接検証した。通常のtarget compatibilityは変更せず、
`phase54-research` buildに限ってexact gfx1030でOCP E4M3 block16を許可し、同一source／compiler／binaryでproductionの
per16 scale recipeを使うE4M3とE5M2を比較した。対象はV620 `GPU-76a08c022586fed6`、ROCm 7.14、binary SHA-256
`00153d65325efefd524409d7d4b8e06a9987161cda68452e97f91019df6b410c`である。各processは5 warmup後に21回をHIP eventで測り、
E4→E5とE5→E4を交互にした5 paired runを実行した。全20 reportが数値oracle、HIP-only、fallback 0、terminal-zero cleanupをPASSした。

| provider／KV長 | E4M3 median | E5M2 median | E5M2 / E4M3 |
| --- | ---: | ---: | ---: |
| short／31 | `235.686 us` | `240.325 us` | `1.0193` |
| short／32 | `242.524 us` | `246.484 us` | `1.0163` |
| short／33 | `248.925 us` | `253.123 us` | `1.0182` |
| short／128 | `228.083 us` | `233.484 us` | `1.0235` |
| short／287 | `444.286 us` | `452.765 us` | `1.0205` |
| short／1023 | `1549.781 us` | `1572.660 us` | `1.0153` |
| long／1023 | `1552.300 us` | `1579.778 us` | `1.0190` |
| long／1024 | `692.009 us` | `717.649 us` | `1.0371` |
| long／1025 | `697.930 us` | `723.569 us` | `1.0365` |
| long／4096 | `2720.073 us` | `2823.916 us` | `1.0379` |
| long／8192 | `5430.905 us` | `5631.468 us` | `1.0371` |
| long／16384 | `10843.410 us` | `11257.743 us` | `1.0383` |

short 6 caseの幾何平均ではE5M2が`1.885%`遅く、paired-run幾何平均の中央値は`1.883%`、範囲は
`1.702–1.987%`だった。long 6 caseではcase幾何平均で`3.431%`、paired-run幾何平均の中央値で`3.430%`遅く、
範囲は`3.333–3.534%`だった。ただしこのv1測定後のレビューで、両decoderがformat-neutralなsign／exponent／mantissa分解と
`ldexpf`を使い、E5M2のFP16との同型性を利用していないことを確認した。従ってv1は未最適化実装のbaselineとしてだけ保持し、
format固有の速度仮説を棄却する証拠には使わない。raw reportはrepository外
`external:phase54/gfx1030/e4-e5-speed-v1/{short,long}/`に保存した。

### format-aware decoderによる訂正測定

OCP E5M2のsign 1 bit、exponent 5 bit、mantissa 2 bitはIEEE binary16とsign／exponent biasが同一なので、全byteを
`uint16_t(bits) << 8`でFP16 bit列へ厳密に写し、gfx1030の`v_cvt_f32_f16`でFP32化するresearch経路を追加した。E4M3側も
公平な比較のため、normal値はsign、biasを7から127へ直したexponent、mantissaからFP32 bit列を直接構築し、subnormalと固有NaN
だけを別処理するformat-aware経路へ変更した。通常buildのdecoderは変更していない。

同じV620、ROCm、shape、5 warmup＋21 sample、5交互paired runで再測定した。binary SHA-256は
`afd3b03eeae82f8e420a193711fb1cac4a120c936f985c5475153d8e22c4fead`である。全20 reportが数値oracle、metadata、HIP-only、
fallback 0、terminal-zero cleanupをPASSし、v1とv2の全20対応reportでper-case output SHA-256も一致した。

| provider／KV長 | E4M3 median | E5M2 median | E5M2 / E4M3 |
| --- | ---: | ---: | ---: |
| short／31 | `234.200 us` | `221.320 us` | `0.9450` |
| short／32 | `240.559 us` | `227.359 us` | `0.9451` |
| short／33 | `247.079 us` | `233.760 us` | `0.9461` |
| short／128 | `224.879 us` | `216.519 us` | `0.9628` |
| short／287 | `442.798 us` | `418.719 us` | `0.9456` |
| short／1023 | `1539.114 us` | `1446.674 us` | `0.9399` |
| long／1023 | `1539.395 us` | `1456.036 us` | `0.9458` |
| long／1024 | `645.561 us` | `561.640 us` | `0.8700` |
| long／1025 | `651.081 us` | `567.121 us` | `0.8710` |
| long／4096 | `2531.398 us` | `2197.593 us` | `0.8681` |
| long／8192 | `5047.109 us` | `4380.145 us` | `0.8679` |
| long／16384 | `10083.213 us` | `8748.517 us` | `0.8676` |

short 6 caseの幾何平均ではE5M2が`5.259%`速く、paired-run幾何平均の中央値は`5.227%`、範囲は
`5.173–5.552%`だった。long 6 caseではcase幾何平均で`11.870%`、paired-run幾何平均の中央値で`11.925%`速く、
範囲は`11.842–11.995%`だった。全case／全pairでE5M2が速い。従ってユーザー指摘どおりv1の方向は未最適化decoderによる
artifactであり、format-awareな現行candidateではE5M2のFP16同型性がV620 attentionで明確な速度優位になる。

この測定はcausal-attention kernelのdevice時間だけを比較し、append量子化、session overhead、全model throughputは含めない。
format-aware decoder自体もresearch feature限定である。次の採用判断は同じV620／model／datasetによるfull-model品質と、appendを含む
end-to-end decode throughputを交互paired runで取得して行う。raw reportはrepository外
`external:phase54/gfx1030/e4-e5-speed-v2-optimized/{short,long}/`に保存した。

[対応する保存済み計画](../../../../plans/archive/2026/08/21-31/phase54-kv-fp8-block16-accuracy-research.md) /
[全体計画](../../../../plans/main-plan.md)
