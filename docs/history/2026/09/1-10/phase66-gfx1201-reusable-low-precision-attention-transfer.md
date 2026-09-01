# Phase 66 gfx1201 reusable low-precision providerとattention移植履歴

## 結果

exact R9700 `gfx1201`で、低精度matmulのprepared provider契約、MXFP8 N128 direct-both、weight形式から独立した
typed causal-attention候補を実装した。MXFP8 matrixはkernel ID 37を測定済みshapeへ限定採用し、attentionの
q4k4／q4k8／q8k8候補は数値完全一致だが同期性能で遅いためproduction不採用とした。

同じprovider境界をMXFP6、NVFP4 W4A16／W4A4、MXFP4 W4A4へ接続し、BF16 weightでも同じattention selectorによる
実dispatchを確認した。MXFP6とNVFP4はoperatorとreviewed full-model、MXFP4は対応済みoperator範囲まで実機実行した。
persistent BF16/FP32 weight展開、persistent FP32 attention/KV plane、cross-request activation cacheは追加していない。

## 実装した共通契約

- `low_precision_matmul_provider.hpp`へ、exact target／architecture、weight・activation format、block/scale、layout、
  activation pack、tile policy、inner product、FP32 accumulate、BF16 RNE outputを型付き契約として分離した。
- 対象formatはOCP MXFP8 E4M3 W8A8 block32/E8M0、MXFP6 E3M2 W6A6 block32/E8M0、
  NVFP4 W4A16／W4A4 block16＋tensor scale、MXFP4 W4A4 block32/E8M0である。NVFP4とMXFP4を
  同一packing／scale semanticとして扱わない。
- public runtimeはprepare時にproviderと具体kernel variantを一度決め、execute、dispatch監査、workspace検証で同じ
  frozen identityを使う。環境変数をprepare後に変更してもplanのprovider identityは変わらない。selectorはmodel名、layer番号、
  prompt、token、benchmark case名、測定結果を受け取らない。
- Phase 62 codecへMXFP4 E2M1/E8M0 block32 viewを追加した。NVFP4／MXFP4のK tailは既存device kernelのceil blockと
  masked loadに合わせてproviderが受理し、MXFP8／MXFP6のK非32倍は従来どおりfail-closeする。

## MXFP8 matrix実証

Phase 65のID 36 `128x64x32` activation/weight direct-loadをcontrolに、同じarithmetic treeを128 output列へ広げる
ID 37 `matmul.mxfp8.w8a8.gfx1201.wmma128x128.bdirect.v1`を追加した。device symbolは
`sllm_mxfp8_w8a8_gfx1201_wmma128x128_bdirect_v1`である。

operatorはM=`127/128/129`、N=`64/127/128/129/256/512/1024`、wide/down production shapeを含め、
独立oracle、repeat digest、HIP-only、fallback false、cleanup 0をPASSした。K31／33は期待どおりhost rejection、K32は実GPUで
受理・実行し、fail-closeをGPU PASSへ読み替えていない。代表同期中央値は次のとおりだった。

| shape | ID 36 | ID 37 | 判定 |
| --- | ---: | ---: | --- |
| M128 K2560 N9216 | 181,641 ns | 155,402 ns | ID 37が14.45%短い |
| M128 K9216 N2560 | 398,403 ns | 373,404 ns | ID 37が6.27%短い |

ID 37はexact `gfx1201`、Phase 65 direct-both family、M%128=0、M>=128、K>=2,048、K%32=0、
128<=N<=16,384、N%128=0へscoped default採用した。N64、tail、small-K、vocabulary、別targetは既存providerへ戻す。
resourceはLDS 1,024 byte、private segment 0、SGPR 40、VGPR 164、spill 0、wave32、static WMMA 16命令だった。

ID 37は互いに独立したoutput列のtile幅だけを変更し、各outputの項、FP32加算tree、scale適用、BF16 RNE stageを変えない。
operator BF16 digestもID 36と一致したため数値分類はN0である。量子化recipe、weight/KV default、quality gateは変更しない。

special-value M128 K2048 N2048はID37で`109,403/127,083/93,282 ns`の3 repeatを実行した。weight value／scale／
BF16 output SHA-256は順に`28115b7a22433e3157e5528d7671cf1d771d280b202be078841199b521558db9`、
`a050683b3a1227b0ad4c085acee3ac4bd90259f784b65fe1acc4c6fd1983b910`、
`dd6ce055b912fb96a003e873cb121a02f80de3e29016fd3efd060494716c137d`だった。最大absolute errorは`2016.0`、
最大relative errorは`0.0004885197849944234`。E4M3 subnormal／tie／max／saturation、E8M0 minimum／finite／NaN scale、
signed zero、Inf／NaNを含み、expected／actual nonfinite `4/4`、mismatch 0、special encoding contract PASSだった。

## common causal attention候補

FP16 KVとstandard OCP MXFP8 E4 KVをtyped load codecで扱うq4k4、q4k8、q8k8 prefill候補を実装した。
selector keyはexact target、query/KV head数、head dimension、query count、committed KV length、KV encoding、
`sliding_window`、明示`score_scale`だけで、model identityとweight encodingは使わない。unsupported window／scaleは既存providerへ
fail-closeし、全control/candidateのoutput digestは一致、最大absolute errorは0だった。

| KV | M | 既存provider | Phase 66候補 | 差 |
| --- | ---: | ---: | ---: | ---: |
| FP16 | 128 | 197,881 ns | 209,242 ns | +5.7% |
| FP16 | 512 | 1,187,928 ns | 1,308,208 ns | +10.1% |
| FP16 | 2,048 | 12,885,318 ns | 15,514,969 ns | +20.4% |
| MXFP8 E4 | 128 | 196,282 ns | 204,803 ns | +4.3% |
| MXFP8 E4 | 512 | 1,192,248 ns | 1,278,049 ns | +7.2% |
| MXFP8 E4 | 2,048 | 13,181,842 ns | 16,776,429 ns | +27.3% |

primary 6行すべてで遅いためproduction selectorへ採用せず、`SLLM_CAUSAL_ATTENTION_PHASE66_TILED_PREFILL=1`の
明示候補として隔離した。M127のq4k1 controlはscalar baselineより速かったが単一境界だけで、primary既存q4k1との
採用根拠にならない。再検討は新しいload/reduction構造がM=128/512/2,048を同期測定ですべて改善し、
persistent FP32 attention/KVを要求しない場合とする。

reviewed Gemmaのq16／kv8とQwen3.5 MoE MXFP4のq16／kv2は、同じtyped selectorがhead geometryを評価した上で候補を
明示非選択した。これはmodel/weight名による分岐ではなく、対応済みq16／kv4 candidate以外を既存providerへ戻すfail-close
evidenceである。

## 他形式への移植

### MXFP6

block32/E8M0、activation pack、prepared providerをMXFP8と共有し、E3M2 codec／decoded inner productを形式固有にした。
M17 K2560 N32 operatorはproduction tiled16 ID 25 `138,641 ns`、baseline ID 21 `37,120 ns`でdigest一致だった。
small-Nではbaselineが速いが、full M=2,048をbaselineへ強制するとlaunch configuration上限を超えたため、現時点では
tiled16をproduction providerとして維持し、small-N専用selectorを後続候補にする。

Qwen3.5-4B MXFP6、2,048 input、FP16 KV、1 warmup＋3 measuredは
`300.943/303.996/301.984 tok/s`、中央値`301.984`、MAD`1.041`だった。resident `4,061,763,072` byte、
peak `5,261,350,400` byte、HIP-only、fallback false、cleanup 0である。

### NVFP4

W4A16は共通provider planから既存row8 tiled device kernelを選ぶ。M32 K2048 N6144は`1,210,092 ns`対baseline
`2,891,391 ns`、M32 K6144 N2048は`1,199,054 ns`対`2,629,668 ns`で、それぞれ58.2%／54.4%短かった。
最大相対誤差は0.0039未満、fallbackなしだった。

W4A4は共通provider routingを既存packed device kernel ID 11へ接続した。K=`15/16/17/31/32/33`とM>1をPASSし、
代表M32 K32 N33は`20,641 ns`、最大相対誤差`0.00380045`だった。同期device A/Bに別candidate kernelは存在せず、
Phase 66で採用したのはprepare時のtyped routingとtail契約である。

reviewed Gemma 4 mixed NVFP4 W4A4／FP8 W8A8 artifact、512 input、1+3は
`20.5576/20.4992/20.4858 tok/s`、中央値`20.4992`、MAD`0.0134`だった。resident `9,201,218,276` byte、
peak `15,271,004,900` byte、HIP-only、fallback false、cleanup 0、process drop後0を確認した。

### MXFP4

MXFP4 E2M1 block32/E8M0を共通codec/providerへ追加し、既存decode ID 14／prefill ID 15へroutingした。
synthetic K=`31/32/33`、M=`1/3/7`と、固定Qwen3.5-35B-A3B MXFP4 artifactのexpert gate/up/down、M>1を
独立oracleへ照合し、最大相対誤差0.00389未満、HIP-only、fallback false、cleanup 0でPASSした。

NVFP4 W4A4と同様に、同期A/B用の別device kernelを追加したのではなく、既存kernelをtyped prepared providerへ移した結果である。
full MoE production routeの性能採否はPhase 66のscope外で、operator範囲のscoped adoptionだけを主張する。reviewed full-MoE
benchmarkが明示的にscopeへ入り、同一artifactの同期baseline/candidateを得られる場合に再検討する。

### BF16 attention

固定Qwen3.5-4B BF16、512 input、Phase 66 attention候補、1+3は
`5,886.223/5,863.158/5,875.651 tok/s`、中央値`5,875.651`、MAD`10.572`だった。resident
`8,411,592,192` byte、peak `8,750,220,800` byte、HIP-only、fallback false、cleanup 0である。
これによりattention selectorがMXFP8 weightへ結合せずBF16 weightでも実dispatchすることを確認したが、operator性能判定により
attention候補自体はproduction不採用である。

## MXFP8 final full-model

Qwen3.5-4B／9B MXFP8、明示FP16 KV、input=`512/1,024/2,048/4,096`、最大4 output、3 warmup＋10 measuredを
exact R9700で実行した。全runは生成token `[23066,23066,23066,23066]`、HIP-only、fallback false、cleanup 0、
process drop後0だった。

| model | input | median tok/s | MAD tok/s | resident | peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-4B | 512 | 3,840.804836 | 11.818178 | 4,954,035,712 | 5,292,664,320 |
| Qwen3.5-4B | 1,024 | 3,806.640973 | 8.471334 | 4,954,035,712 | 5,579,650,560 |
| Qwen3.5-4B | 2,048 | 3,767.237995 | 6.802385 | 4,954,035,712 | 6,153,623,040 |
| Qwen3.5-4B | 4,096 | 3,249.069405 | 5.665339 | 4,954,035,712 | 7,301,568,000 |
| Qwen3.5-9B | 512 | 1,988.722356 | 4.716409 | 11,205,394,944 | 11,621,093,888 |
| Qwen3.5-9B | 1,024 | 2,231.573186 | 3.767540 | 11,205,394,944 | 11,985,150,464 |
| Qwen3.5-9B | 2,048 | 2,261.647647 | 11.942724 | 11,205,394,944 | 12,713,263,616 |
| Qwen3.5-9B | 4,096 | 2,069.842794 | 11.873115 | 11,205,394,944 | 14,169,489,920 |

この測定は同じmodel非依存selectorが4B／9Bの実shapeへ適用される証拠であり、別architecture、別artifact、別R9700、
別software tupleへ一般化しない。request workspace arena high-waterは`1,080,836,096` byte。外部HBM/GTTはこのfinal seriesで
別samplingしておらず、allocator auditのprocess drop後0を資源復帰証拠とする。

## profile、identity、最終gate

final immutable CLIのQwen3.5-4B MXFP8、2,048 input、FP16 KVをrocprofv3で1 warmup＋3 measuredした。
long embeddingから最初のargmaxまでをprefill区間として切り出すと、1評価の全kernel duration平均は`532.269 ms`、
中央値は`532.570 ms`だった。trace span中央値は`546.157 ms`、serialized kernel sumとの差は`13.649 ms`である。
この差はqueue idleとprofiler／host effectsを含み、launcher codeだけへ帰属できる値ではない。profile下のmeasured
throughput中央値は`3,755.005 tok/s`、対応する非profile 3+10中央値は`3,767.237995 tok/s`だった。

| 区分 | dispatch/評価 | duration/評価 | 全kernel比 |
| --- | ---: | ---: | ---: |
| MXFP8 ID37 main aligned | 200 | 262.157 ms | 49.25% |
| small-N/fallback matrix＋LM head | 49 | 22.437 ms | 4.22% |
| activation quantization | 248 | 21.841 ms | 4.10% |
| causal attention prefill | 8 | 109.178 ms | 20.51% |
| GDN recurrent本体 | 24 | 66.613 ms | 12.51% |
| elementwise | 104 | 11.618 ms | 2.18% |

raw trace／profileはrepositoryへ含めない。kernel trace CSV SHA-256は
`2bd1efbf56d119239da62ffcede65247650bab683b74636081bd315681df12d9`、kernel stats CSVは
`d8a3966dc6446781af1bd846f2fd6ed6cc893c1892e17c16a3a7b36c34848f9e`である。

final CLI／matrix runner／attention runner SHA-256は
`55fd3a5e7fd85f9685739964d7128007d64195e5f5983580523138272044dc9c`、
`030168e04610dc1b6d2ba5c81ac1ad6f07355a1ef145620063fb110d16bd86ea`、
`c4643ccd55cf3ad115c7fd91908fa60f533ea33e9b406412d5fdbaec1744fd47`である。matmul host object、embedded fatbin、
extracted exact gfx1201 code objectは`3ada513d756ac5364896fb776cc655d50cad35752e2b0e2700c9dde434dd4ad5`、
`78787b13891d2afb6781963524429eb37d84db3dcd8d2d2a8b8f91cd20f91e99`、
`4adc1528dddb2c98a564cd3a334c5b36203e45581dc0e89b1ff3a89b9dda8a88`である。
整形とincremental dependency修正後に再ビルドしても6 identityは不変だった。public-runtime frozen-provider host test、
gfx1030／gfx942 real HIP compile-only、codec実GPU oracle、Rust/C++ format、JSON/schema/H3/G2/P0 source closure、
markdown local link、`git diff --check`をPASSした。compile-onlyを当該GPUの実行PASSとは扱わない。

## provenance

Phase 66の実装に第三者codeはreuseしていない。Phase 65で固定したllama.cpp／SGLang等の比較は、target/format/shape別provider、
consumer向けactivation layout、複数tile familyという抽象所見だけに限定した。Q8/Q8_1の式、layout、tile table、source expression、
symbolは移植せず、Phase 66のcodec、selector、kernel、oracleはsLLMの既存実装とOCP format契約から作成した。

[追跡済み要約](../../../../../ci/matrix/phase66-gfx1201-low-precision-provider-summary-v1.json) /
[Phase 66保存済み計画](../../../../plans/archive/2026/09/1-10/phase66-gfx1201-reusable-low-precision-attention-transfer.md) /
[Phase 37以降のロードマップ](../../../../plans/active/2026/08/21-31/phase37-plus-mi300x-and-llama-gap-roadmap.md) /
[比較・provenance境界](../../../../provenance/phase65-inference-engine-comparison.md)
