# Phase 63 gfx1201向け大規模MXFP matrix prefill 履歴

## 2026-09-01: 新Phaseとして計画

- ユーザー指示により、Phase 62後のMXFP prefill性能残差をPhase 63へ割り当てた。
- exact R9700 `gfx1201`、固定Qwen3.5-4B、明示FP16 KVの追加実測では、MXFP8既定／col8／BF16 controlの
  prefill中央値が512 inputで`107.999／281.991／5,950.932 tok/s`、2,048 inputで
  `105.011／274.853／5,483.034 tok/s`だった。既存1,024-token固定列でもcol8／BF16は
  `279.115／5,941.838 tok/s`で、反復token列固有の結果ではなかった。
- 全試行はexact `gfx1201`、HIP-only、fallbackなし、同一provider内token再現、request/session cleanup 0だった。
  MXFP8 artifactは`sha256:f253d9f47603d84718b4fdb898b434e493d732b52838ba9abfdfafe73a5d076f`、
  binaryは`sha256:f52462bce66a08b24083cc9539fb7c907472fb6429c158d32f9ee4894f6e1d76`である。
- 専用read-only subagentがsLLMとローカルSGLang実装を比較し、source／疑似コード／tile値／symbolを返さず、
  大規模matrix provider、shape別prepared selector、activation量子化共有、projection family分離、launch二次要因という
  抽象所見だけを親agentへ渡した。
- Phase 63はSGLang非copyのclean-room laneとし、sLLM既存codec、OCP MX v1.0、AMD公開資料、独立oracle／実測から
  exact `gfx1201`向けモデル非依存MXFP8 matrix providerを設計する。1,000 tok/sは性能目標であり一律hard gateにしない。
- この時点では計画文書だけを変更し、production source、provider、artifact、binary、default、public ABIは変更していない。

## 2026-09-01: provider実装とselector固定

- public ABIの予約済み末尾へadditive kernel ID 30とstable symbol
  `matmul.mxfp8.w8a8.e4m3.block32.prefill.wmma128x16x32.v1`を追加した。OCP MXFP8 E4M3 valueとE8M0 block32 scaleを
  常駐表現のまま読み、K32を2個のFP8xFP8-to-FP32 WMMA K16へ分け、block scale適用後にFP32 accumulateしBF16 RNEへ出す。
  whole-layer/model展開、persistent FP32 plane、request間cacheは追加していない。
- prepared selectorはexact target、encoding、M/N/Kだけで判定し、model名やpromptをkeyにしない。production scopeは
  exact `gfx1201`、M>=128、K>=2,048、2,048<=N<=16,384、K%32=0、N%16=0である。M=1、M=127、narrow、LM-head-like、
  exact `gfx1030`／`gfx942`／未知targetは既存row/decode providerへ戻す。prepare後の環境変更は保存variantを変えない。
- env overrideはbaseline、明示row8、MMQ/tiled16、明示WMMA、scoped defaultの順に解決する。host testはgfx1030/gfx942/unknown、
  M=`127/128/129`、wide/narrow/LM head、force優先順、prepare-time不変性をPASSし、両非採用targetもcompileした。
- Qwen実行監査へkernel ID/symbol別のaccepted dispatch histogramを追加し、品質runnerはrow8でID30=0、candidateでID30>0を
  primary/repeatの両方でfail-closeする。worker deadline、kill/wait、stderr継承も追加した。

## 2026-09-01: exact GPU／ISA／数値検証

- 最終CLI `sha256:6592831f4569817bb3d274399460eb011d2d977eea4812621b0b5d8d06409e7c`、operator runner
  `sha256:3b97dc53d1777ff9f0ca5d037238490890bd525258b9228e3043c63f9bf33a5f`、quality runner
  `sha256:eaa068866084bc9d02bf285b5f246a1b4a26b5af35bc6cf1352ff38d8c32d6f4`を固定した。
- CLIから再抽出したexact gfx1201 code objectは
  `sha256:ae8b86c90c08d63c7818591e858b2cf193e3eaf0a37ad6df362bbb653d4e95f3`で、対象symbolに
  `v_wmma_f32_16x16x16_fp8_fp8`を2命令確認した。wave32、WG256、LDS 13,376 byte、SGPR 34、VGPR 52、spill/private 0である。
- operator report `sha256:0becfa1d13017c7a4390d4d155ca0cfb9d9046c0705730f67c5ed60aac8ad244`は24 caseを
  3 repeatした。candidate 6 case／18 submission、M=`127/128/129`、wide/down/output/narrow、M=1非選択を含み、
  production最大相対誤差`0.0036960265`だった。special caseはE4M3 subnormal/tie/max/saturation、E8M0 min/finite/NaN、
  signed zero、Inf/NaN byteを直接検査し、非有限4/4一致、relative `0.0004885198`、repeat digest一致だった。
- 同じMXFP8 artifactをrow8とscoped candidateの隔離processで比較したquality report
  `sha256:996c3e1f2352c3fbd72473ac81ebc46e91cad1ea4a28b98c32008553bd302bf9`はPASSした。ID30 dispatchは
  `0／2,208`、top-1 `19/20=0.95`、KLD mean `0.0030239021`、p99/max `0.0131994619`、最大logit差`0.625`、
  perplexity相対差`+1.517253%`だった。最初のlogit差は`b255` prefill position 254、最初のtoken差は`b512` decode
  position 512の`278→220`。両providerのrepeatはbitwise一致し、fallback false、cleanup 0だった。
- 実数式、E4M3/E8M0入力、FP32 accumulation、BF16 RNEを維持し、K32の逐次FP32和を2個のK16 FP32 WMMAと
  block-scale後のFP32和へ変えた。項やscaleを欠落せず依存深さも増えないため数値台帳N1へ分類した。top-1 `0.95`は
  品質観測であり、旧KV default判定の`0.99` thresholdをこのW/A providerへ流用していない。

## 2026-09-01: full-model性能、profile、採否

- 最終CLI、固定Qwen3.5-4B MXFP8 artifact、明示FP16 KV、3 warmup＋10 measuredの既定経路で、prefill中央値／MADは
  512 `939.105/2.036`、1,024 `953.913/3.905`、2,048 `947.248/1.294`、4,096 `908.266/1.571 tok/s`、
  4,096 chunk 2,048は`899.118/1.220 tok/s`だった。全runはHIP-only、fallbackなし、cleanup 0である。
- 同一最終binaryの1 warmup＋3 measured controlはcol8が512/1,024/2,048/4,096で
  `281.961/277.554/272.433/268.358 tok/s`、BF16が`5,929.065/6,010.538/5,476.576/4,387.050 tok/s`。
  candidateはcol8比3.33〜3.48倍で、測定最大sampleは`959.623 tok/s`だった。1,000 tok/s目標は未達として残す。
- rocprofv3の2,048 input、1 warmup＋3 measuredはprefill `967.401/965.474/963.509 tok/s`だった。kernel-duration sumの内訳は
  WMMA 73.02%、残存row8 11.73%、activation quantization 1.06%、attention 6.53%、linear attention/GDN 3.82%、
  elementwise/norm/other 3.84%。量子化共有／fusionの条件は満たさず実施しない。次の残差はWMMA+row8のmatrix 84.75%である。
- row8/candidateのmodel residentは共に4,954,035,712 byte、request state 68,354,048 byte、workspace 270,738,432 byte、
  total 5,293,128,192 byteだった。終了後process 0、HBM/GTT `57/59 MiB`、ECC error 0を確認した。
- 採否はexact gfx1201 production shapeの`scoped-default`。rollbackは明示row8またはscope外の自動row8である。MXFP6、NVFP4、
  gfx1030、gfx942、別SKU／software tupleへ一般化しない。SGLangからsource、疑似コード、tile値、symbolを受け取らない
  clean-room境界を維持し、integration reviewのcorrectness/security/ABI/selector/dispatch blockerは0だった。
- 追跡済み要約は
  [`phase63-gfx1201-mxfp8-wmma-prefill-v1.json`](../../../../../ci/matrix/phase63-gfx1201-mxfp8-wmma-prefill-v1.json)を正本とする。

## 2026-09-01: contribution LDS削除とN tile再設計

- ユーザー指示でPhase 63を再オープンし、既知のsLLM内部残差を先に実行した。SGLangは再調査せず、v1 CLI、operator、profileを
  固定baselineとして、contribution LDS、N tile、K pipeline、row8 shape残差を独立候補にした。
- rocWMMA accumulatorをpublic `apply_data_layout<row_major>`で一度だけlane-local row-majorへ変換し、v1がK32ごとに使った
  8,192-byte FP32 contribution LDS store/readとそのbarrierを削除した。単純なfragment index直読はlane mappingが一致せず数値FAIL、
  各K16 MMA後のrow-major accumulator変換は正しいが遅く、2個のK16 MMA後に一度だけ変換する候補を採用した。
- N16／N32／N64を同じoracleで比較した。N64はM128/K2560/N9216を約`550 us`、N32は約`686 us`、N16は約`886 us`で処理した。
  N64は一部down/output shapeでN16より遅いが、Qwen full-modelのwide gate/up比重が大きく、2,048 inputのdraft中央値を
  N16 `1,001.657`、N32 `1,358.049`、N64 `1,473.987 tok/s`へ伸ばした。
- N=1,024をN64へ強制するとoperatorは約`0.58 ms`で、従来row8約`1.22 ms`の半分以下だった。N=32はN64約`0.59 ms`に対し
  row8約`0.057 ms`で約10倍遅いためrow8を維持した。vocabulary N=248,320はselector上限外かつM=1 decodeのため拡張しない。
- productionはkernel ID 31、logical symbol `matmul.mxfp8.w8a8.e4m3.block32.prefill.wmma128x64x32.v2`として追加した。
  ID 30 `wmma128x16x32.v1`は明示controlへ残し、既存baseline／row8／MMQ／tiled16 overrideを維持した。v2 selectorはexact
  `gfx1201`、M>=128、K>=2,048、1,024<=N<=16,384、K%32=0、N%64=0だけを選ぶ。

## 2026-09-01: K pipeline候補、最終GPU証跡、closeout

- ping-pong LDSでK32終端barrierと次block loadを重ねる二重buffer候補を実装・測定した。数値はPASSしN=1,024で約6.6%改善したが、
  wide M128/M129は約4%、K=2,048/N=2,048は約8%遅化し、2,048 full-modelは`1,756.549→1,763.702 tok/s`の
  約0.4%に留まった。LDS倍増に見合う一貫した利得がないため単一bufferへ戻して棄却した。
- 最終CLI `sha256:0e44f19142814d2b93fc811fee85d394fa2f894c8db87a92ac681cfcc090138a`、operator runner
  `sha256:77dbe38194274078797b4b782bab02e7781ee9a4d687bec402538c1a77336c89`、quality runner
  `sha256:07712265464081ea9d79acffbbcf293e279ea82765de3174d943b81c91376813`を固定した。抽出matmul code objectは
  `sha256:0acd6d364d4c4d3c0e33a3c479ed4e804cf8f498023079d5ddaf2a65c1f926b7`で、v2 symbolはWMMA 8命令、wave32、WG256、
  LDS 6,912 byte、SGPR/VGPR 33/103、private/spill 0だった。
- 最終operator report `sha256:640742935881ea8b42a5bd4e113e152c10c88d31559eedf795a71876c5314077`はcandidate 7 case／
  21 submissionを3 repeatした。最大production相対誤差`0.0036960265`、special `0.0004885198`、全repeat output digest一致、
  HIP-only、fallback false、cleanup 0である。
- 同一artifact provider品質report `sha256:362c34e3d007500952fa3be947d7a86007f9f9731d9fa769448ca4ce5d192152`は
  row8／v2を隔離processで各2回実行し、ID31 dispatch `0／2,400`、top-1 `19/20=0.95`、KLD mean `0.0029974001`、
  p99/max `0.0153089212`、最大logit差`0.5859375`、perplexity相対差`-0.516618%`、repeat bitwise一致、cleanup 0だった。
- 3 warmup＋10 measuredのprefill中央値／MADは512 `1,727.595/1.291`、1,024 `1,814.619/3.388`、
  2,048 `1,722.844/6.018`、4,096 `1,588.366/2.279 tok/s`、4,096 chunk 2,048 `1,573.875/5.052 tok/s`。
  全長でv1を約1.75〜1.90倍上回り、1,000 tok/s観測目標も超えた。全runはHIP-only、fallback false、cleanup 0だった。
- 2,048 profileはprefill中央値`1,743.802 tok/s`、kernel-duration sumでWMMA v2 71.96%、row8 1.55%、activation quantization
  1.78%、causal attention 8.66%、linear recurrent 5.12%。PMCsはrocprofv3がrequested gfx1201 counterを0で返したため、
  counter由来の性能claimを行わずISA/resource/profileを採否根拠とした。
- host selector/prepare-time contract、gfx1030 wave32とgfx942 wave64のrelease compile-only、sllm-hip focused bin 6件、
  sllm-core unit 529件、format／JSON／Markdown local link検査をPASSした。production採否を`scoped-default`としてPhase 63を
  再closeした。persistent BF16/FP32 weight展開、FP32 attention/KV plane、数値recipe、MXFP8 E4 KV default、明示FP16 rollbackは
  変更していない。追跡済み正本は
  [`phase63-gfx1201-mxfp8-wmma-prefill-v2.json`](../../../../../ci/matrix/phase63-gfx1201-mxfp8-wmma-prefill-v2.json)である。

## 2026-09-01: Qwen3.5-9B MXFP8へのmodel共通経路確認

- 固定Qwen3.5-9B revision `c202236...`からOCP MXFP8 E4M3 W8A8 GGUF
  `sha256:eff9e54d04056f968604301a856bd871454913a24b3fb26cda61ce178d7c4033`を生成し、同じR9700、CLI、FP16 KV、
  direct inputでPhase63既定WMMAと強制row8を各1 warmup＋3 measuredした。
- 512／1,024／2,048／4,096 inputのWMMA中央値は`788.346/896.405/912.870/889.445 tok/s`、row8は
  `55.272/54.919/54.897/54.671 tok/s`で、WMMAは`14.26/16.32/16.63/16.27x`だった。residentは両providerとも
  `11,205,394,944 byte`で、最大peakは4,096 inputの`14,169,489,920 byte`。全runはHIP-only、fallback false、cleanup 0だった。
- 512-token kernel traceはID 31 `wmma128x64x32.v2`を800 dispatch確認した。provider内の3 repeat token列は一致し、4長中3長は
  provider間も一致した。1,024 inputだけ4番目の生成tokenがWMMA `16`、row8 `17`へ分岐した。これはPhase63 N1の文脈へ置く
  観測であり、9Bの包括的品質suiteを実行済みとは扱わない。追跡済み要約は
  [`phase63-gfx1201-qwen35-9b-mxfp8-prefill-v1.json`](../../../../../ci/matrix/phase63-gfx1201-qwen35-9b-mxfp8-prefill-v1.json)である。

[全体計画](../../../../plans/main-plan.md) /
[対応する計画](../../../../plans/archive/2026/09/1-10/phase63-gfx1201-mxfp-matrix-prefill.md) /
[Phase 37以降のロードマップ](../../../../plans/active/2026/08/21-31/phase37-plus-mi300x-and-llama-gap-roadmap.md)
