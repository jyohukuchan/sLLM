# Phase 63: gfx1201向け大規模MXFP matrix prefill

## 状態

`完了`（2026-09-01追加最適化をclose）。OCP MXFP8 E4M3 W8A8の大規模prefillを対象に、
model非依存WMMA provider、prepared selector、dispatch監査を実装し、contribution LDS削除、N64 tile、N=1,024拡張を採用した。
operator／ISA／full-model／品質／profile／資源を検証し、数値recipe、model artifact、KV default、persistent BF16/FP32展開は変更していない。

### 2026-09-01 追加最適化の固定baselineとacceptance

Phase 63 v1の最終CLI `sha256:6592831f4569817bb3d274399460eb011d2d977eea4812621b0b5d8d06409e7c`、operator report
`sha256:0becfa1d13017c7a4390d4d155ca0cfb9d9046c0705730f67c5ed60aac8ad244`、2,048 profileのWMMA 73.02%、
row8 11.73%、activation quantization 1.06%を追加作業の固定baselineとする。SGLangは再調査せず、sLLM source、AMD公開API、
exact gfx1201のcounter／oracle／実測だけから独立候補を作る。

- **P63-B0 counter固定**: 現行WMMAのLDS、barrier、occupancy、matrix issue、cache/global load counterを取得可能な範囲で記録する。
  profiler/toolがcounterを提供できなくても、ISA、resource、同期回数、同期済みoperator timingから候補評価を継続できる。
- **P63-B1 contribution経路**: K32ごとのFP32 contribution 8,192-byte LDS storeと同量read、3 barrierを削減する候補を作る。
  accumulator fragmentのlane mappingを独立oracleで固定できない実装、非公開API依存、誤差方向を分類できない候補は採用しない。
- **P63-B2 N tile再利用**: N=16 baselineに対してN=32/64等を独立候補として比較し、同じactivation tileのN workgroup間再ロードを減らす。
  M/N/K境界、LDS/VGPR/occupancy、wide/down/outputのshape逆転を候補ごとに記録する。
- **P63-B3 K pipeline**: load→barrier→WMMA→scaleの直列区間をping-pong LDSまたは複数K32 schedulingで重ねる候補を、
  B1/B2とは別に評価する。barrier削減、tail、scale generation、buffer ownershipをfail-closeする。
- **P63-B4 row8残差**: N=1,024、N=32、vocabulary N=248,320を分離し、WMMAまたは別tileをoperatorで比較する。
  一つのselector scopeへまとめず、勝者がないshapeはrow8を維持する。
- 各候補はexact gfx1201 release build、同一MXFP8 value/scale、FP32 accumulation、BF16 RNEでbaselineと3 repeat以上比較する。
  E4M3/E8M0 special、非有限、M=`127/128/129`、実Qwen shape、fallback、repeat、cleanupをPASSへ要求する。
- 採用候補は同一artifact品質runner、512/1,024/2,048/4,096 full-model、2,048 profile、resident/workspaceを再取得する。
  draftは1 warmup＋3 measured、最終採用は3 warmup＋10 measuredとし、1,000 tok/sは引き続き観測目標でhard gateにしない。
- 候補が遅い、数値N2/N3、resource増、target非分離の場合は棄却理由を残してPhase 63を再closeできる。

## 背景と固定baseline

Phase 61でOCP MXFP8 E4M3 W8A8／MXFP6 E3M2 W6A6をQwen3.5へ統合し、Phase 62でscalar/block codec、
packed I/O、typed view、target別specializationをmatmul、KV、attentionへ共有した。Phase 62後のmulti-column候補は
短いprefillを2倍超改善したが、format／N shapeで勝敗が逆転したためbenchmark-onlyに留めている。

2026-08-31〜09-01に、exact AMD Radeon AI PRO R9700 `gfx1201`、単一GPU、batch 1、明示FP16 KV、
固定Qwen3.5-4B MXFP8 GGUF `sha256:f253d9f47603d84718b4fdb898b434e493d732b52838ba9abfdfafe73a5d076f`、
実行binary `sha256:f52462bce66a08b24083cc9539fb7c907472fb6429c158d32f9ee4894f6e1d76`で、
1 warmup＋3 measuredのdirect benchmarkを実行した。全試行はHIP-only、fallbackなし、同一provider内のtoken列再現、
request/session cleanup 0だった。

| input | MXFP8既定 prefill | MXFP8 col8候補 | 同一model BF16 control |
| ---: | ---: | ---: | ---: |
| 512・反復token列 | 107.999 tok/s | 281.991 tok/s | 5,950.932 tok/s |
| 1,024・既存`prefill-long`固定列 | 未測定 | 279.115 tok/s | 5,941.838 tok/s |
| 2,048・反復token列 | 105.011 tok/s | 274.853 tok/s | 5,483.034 tok/s |

SGLangの別model／別形式であるQwen3.8-27B mixed NVFP4 W4A16/FP8は、同じR9700の単一requestで
2,048-token prefillを1,633.5 tok/sと記録している。これはsLLMとのstrict同条件比較ではないが、read-only比較と
sLLM自身のBF16 controlから、GPU、attentionのFP32保存有無、framework全体ではなく、現行MXFP prefill matmulの
計算構成が第一の性能残差であると判断する。

## clean-room参照境界

- `sLLM.md`と[provenance方針](../../../../../provenance/README.md)に従い、SGLangは技術的事実、制約、評価設計だけを得る
  非copy参照とする。SGLang sourceのcopy、adapt、port、近接した疑似コード化を行わない。
- 専用read-only subagentはsLLMとローカル`../sgl-rdna4-nvfp4`を分離調査し、親agentへコード断片、具体的な
  instruction列、転記可能な式、tile table、SGLang固有symbolを返さず、次の抽象所見だけを返した。
  1. 大規模prefill用matrix-instruction providerの不足。
  2. target／format／M／N／K／projection family別の準備時selector不足。
  3. 同じactivationを使うprojection間の量子化重複。
  4. wide、narrow、LM headを同じproviderへ送ることによるshape別逆転。
  5. 量子化＋matmulの追加dispatchは二次要因であり、prefill Graphだけでは主因を解消しない。
- Phase 63の実装根拠は、sLLM既存semantic／codec、OCP MX v1.0の公開format、AMDの公開HIP／ISA／matrix API資料、
  独立した数値oracleと実測とする。SGLangで使われた具体的なtile値やsource構造を候補の初期値にしない。
- 独立設計の候補値はsLLMのshape sweepから導出し、調査記録とimplementation commitを分離する。第三者source表現との
  類似が疑われる場合は直接reuseとして扱い、このPhaseのclean-room候補には採用しない。

## 目的

exact `gfx1201`で、大きいMのOCP MXFP8 E4M3 W8A8 matmulをRDNA4のmatrix instructionへ送る
モデル非依存providerを追加し、静的shape／layoutとM領域から準備時に選ぶproduction selectorを確立する。
MXFP value／E8M0 scaleを常駐表現のまま消費し、persistentなBF16/FP32 weight展開を作らず、既存の
FP32 accumulation、BF16 RNE graph境界、format semantic、resident memory削減を維持する。

1,024〜2,048-tokenの固定Qwen3.5-4Bで1,000 tok/sを超えることを性能目標として観測するが、これをPhase完了の
一律hard gateにはしない。候補が目標未達でも、正しさ、profile、採否、残差を固定してscoped adoptionまたは棄却まで
完了できる。逆に1点だけ1,000 tok/sを超えても、安定したadoption scopeを証明できなければproduction採用しない。

## 固定acceptance

### 1. providerと数値契約

- primary実装範囲はexact `gfx1201`、OCP MXFP8 E4M3 W8A8、`M>1`のprefill matmulとする。M=1 decode、
  exact `gfx1030`、`gfx942`、未知targetは現行providerを維持し、新providerへ誤送信しない。
- provider interfaceはmodel名を受け取らず、block-scaled tensor view、target capability、dtype/encoding、layout、alignment、
  M/N/K、accumulation/output contractから選択できる形にする。MXFP6とNVFP4は形式差を保持して将来同じmatrix provider境界を
  利用できる設計にするが、このPhaseで両形式のproduction kernel実装を完了条件にしない。
- packed valueとblock scaleはtile内のboundedなregister/LDS作業域だけでmatrix instruction入力へ変換する。
  whole-layerまたはwhole-modelのBF16/FP32展開、persistent FP32 attention/KV plane、request間cacheを追加しない。
- candidateがnative matrix pathを名乗るには、exact release artifactのdisassembly／ISA集計で対象matrix instructionを確認し、
  scalar fallback、別dtype GEMM、CPU fallbackをPASSへ数えない。
- OCP value byte、E8M0 scale byte、block 32、RNE/saturation、NaN/Inf semantic、K非32倍のfail-closeを変更しない。
  real-number equation、積・scale適用順、accumulation tree、丸めstageを文書化し、数値変更台帳のN0/N1/N2/N3へ分類する。
  N2はユーザー判断前にproduction採用せず、N3は採用しない。

### 2. shape selectorとadoption scope

- selector keyはexact target、encoding、layout/alignment、M領域、N/K、projectionのshape familyだけで構成する。
  model名、benchmark case名、prompt、token ID、実測後の結果をkeyにしない。
- selectorはmodel／graph準備時に固定し、tokenごとのhost判断やhot loop内のformat／shape分岐を追加しない。
- large-M matrix候補、Phase 62 col4/col8、既定row/tiled providerを同じsweepで比較する。wide projection、narrow補助projection、
  output/down projection、LM headを一つのtileへ強制しない。
- 採用するMまたはshape境界は最終測定前にmanifestへ固定し、各境界の`B-1/B/B+1`と範囲内の複数代表点を実GPUで確認する。
  範囲外は既存providerへ明示的に補完し、未知shapeをcandidateへ広げない。
- 採否はmain planの`adoption scope S`規則に従い、shared／scoped／benchmark-only／rejectedを候補ごとに記録する。
  固定改善率や全shape一律非悪化を新しいhard gateにしない。

### 3. 数値・correctness

- host側の独立dequantized oracleと実GPUcandidateを比較し、FP32 accumulator／BF16 outputに対するop別toleranceを
  baseline取得時に固定する。既存の一律`0.02`だけへ依存せず、absolute、relative、非有限値、最大誤差位置を記録する。
- format境界`31/32/33`、MおよびN/K tail、zero/subnormal/tie/max/Inf/NaN、実Qwen projection shapeを含める。
  行列サイズは少なくともM=`1/3/17/127/128/129/511/512/513/1023/1024/1025/2047/2048/2049`から、
  provider境界と実行時間に必要な代表集合をmanifestへ固定する。M=1はcandidate非選択の回帰である。
- 同一providerのrepeatは決定的で、padding、tail、unsupported shape、allocation failure、completion publication、cleanupを
  fail-closedに確認する。compile-only、zero test、timeout、CPU fallbackをGPU PASSにしない。
- 固定Qwen3.5-4Bでは現在のMXFP8 providerを数値baselineとし、既存10-case quality runnerでtop-1、KLD、perplexity、
  最初のlogit/token分岐を記録する。token完全一致だけをhard gateにせず、差を数値台帳の分類と結び付ける。

### 4. 性能・資源

- exact `gfx1201`の同一release build、同一MXFP8 artifact、明示FP16 KVで、既定provider、Phase 62 col8、candidate、
  同一model BF16 controlを直列に比較する。candidate residentを完全解放してから次のproviderへ移る。
- full-modelは512、既存1,024-token `prefill-long`固定列、2,048、4,096 inputを対象にし、4,096は2,048-token chunkも測る。
  draftでは1 warmup＋3 measured、採用候補の統合測定では1 warmup＋5 measured以上の中央値とMADを記録する。
- sLLM内部のsubmit→prefill completeによるprefill token/sに加え、SGLangの公開結果と意味を近付ける
  `input tokens / TTFT`、E2E、decode token/s、kernel GPU span、kernel-duration sumを記録する。
- profileは量子化、matrix mainloop、scale処理、attention/GDN、elementwise、host/launch gapへ分ける。大規模matrix provider採用後に
  activation量子化またはlaunchが有意な残差である場合だけ、後述の共有/fusion候補へ進む。
- model resident bytesは現行MXFP8と同一を原則とし、追加workspaceはrequest arena内でboundedにする。workspace／HBM peak、
  dispatch／kernel count、code size、fallback、終了後のHBM/GTT baseline復帰を採否へ含める。
- 1,000 tok/sはexact Qwen3.5-4B／gfx1201の性能目標であり、別model、別GPU、concurrency、production tail latencyへ一般化しない。

### 5. conditional activation共有／限定fusion

- large-M matrix providerを先に実装・profileし、同じBF16 activationを消費する複数projectionの量子化が残るGPU spanの
  materialな割合を占める場合だけ、共有済みactivation viewまたはprojection bundleの限定候補を作る。
- 共有bufferはinput buffer generation、plan identity、liveness、request ownershipを持ち、Phase 62で棄却した
  generationなしのcross-plan cacheを復活させない。N tileごとにactivation量子化を再実行する単純fusionも再提案しない。
- 共有／fusionはmatrix providerの完了条件ではない。profile上の寄与が小さい、workspaceまたはdispatchが悪化する、
  stale readをfail-closeできない場合は棄却理由だけを記録してPhaseを閉じる。

## 作業単位

1. **P63-A0 baseline／clean-room固定**: 上記artifact、binary、software tuple、512/1024/2048 baseline、read-only抽象所見、
   SGLang非copy境界をevidence manifestへ固定する。
2. **P63-A1 provider contract**: model非依存のblock-scaled matrix capability、prepared selector、workspace見積り、
   fallback／unsupported契約をhost testで固定する。
3. **P63-A2 gfx1201 matrix prototype**: MXFP8 block scale semanticを保つ複数の独立候補を実装し、tiny/non-aligned oracle、
   ISA、同期済みoperator timingで候補を絞る。
4. **P63-A3 shape sweep／selector**: Qwen production shapeと境界M/N/Kを測り、stable dispatch key、adoption scope、
   現行providerへ戻す範囲をmanifestへ固定する。
5. **P63-A4 full-model／profile**: 512/1024/2048/4096のMXFP8 default／col8／candidate／BF16 control、数値品質、
   TTFT、prefill、decode、VRAM、dispatch、cleanupを取得する。
6. **P63-A5 conditional residual**: profileが支持する場合だけactivation共有／限定fusionを一候補ずつ評価する。
7. **P63-A6 integration／closeout**: focused host/GPU test、Qwen CLI/direct benchmark、既定MXFP8 KVと明示FP16 KVの回帰、
   numerical ledger、compatibility、main plan、matching historyを同期し、計画をarchiveへ移す。

## 完了結果

- exact R9700 `gfx1201`だけで、M>=128、K>=2,048、1,024<=N<=16,384、K%32=0、N%64=0のMXFP8 E4M3 W8A8を
  kernel ID 31 `wmma128x64x32.v2`へ送る。M=1、M=127、N=32、vocabulary N=248,320、exact `gfx1030`／`gfx942`／未知targetは
  既存providerを維持する。selectorはprepare時に一度固定し、ID 30のN16 providerは明示比較用に保持した。
- v2はK32ごとの8,192-byte FP32 contribution LDS store/readを削除し、rocWMMA accumulatorを一度だけrow-major lane layoutへ変換する。
  同じactivation tileからN16 WMMAを4個生成してN64を処理する。最終CLIは
  `sha256:0e44f19142814d2b93fc811fee85d394fa2f894c8db87a92ac681cfcc090138a`、matmul code objectは
  `sha256:0acd6d364d4c4d3c0e33a3c479ed4e804cf8f498023079d5ddaf2a65c1f926b7`。wave32、WG256、LDS 6,912 byte、
  SGPR/VGPR 33/103、spill 0、対象symbol内のWMMA 8命令を確認した。
- candidate 7 case／21 submissionを3 repeatし、M=`127/128/129`、wide/down/output、N=1,024、N=32非選択、special E4M3/E8M0を
  独立oracleへ照合した。最大相対誤差はproduction `0.0036960265`、special `0.0004885198`、repeat output digest一致、
  HIP-only、fallback false、cleanup 0だった。
- 同一MXFP8 artifactのrow8対v2品質比較は10 case／20 rowでtop-1 `19/20=0.95`、KLD mean `0.0029974001`、
  p99/max `0.0153089212`、perplexity相対差`-0.516618%`だった。旧KV defaultの`0.99` gateは適用せず、real-number equation、
  FP32 accumulator、BF16 RNEを維持する固定WMMA tree変更としてN1へ分類した。
- 3 warmup＋10 measuredのprefill中央値／MADは512 `1,727.595/1.291`、1,024 `1,814.619/3.388`、
  2,048 `1,722.844/6.018`、4,096 `1,588.366/2.279 tok/s`。4,096 chunk 2,048は`1,573.875/5.052 tok/s`だった。
  v1比は各長で約1.75〜1.90倍となり、1,000 tok/s観測目標を全長で超えた。
- 2,048 profileはWMMA v2 71.96%、残存row8 1.55%、activation quantization 1.78%、causal attention 8.66%、
  linear recurrent 5.12%だった。row8残差の大半をN=1,024拡張で解消した。model resident `4,954,035,712 byte`は不変で、
  persistent BF16/FP32 weight展開やFP32 attention/KV planeは追加していない。
- N32はv1より改善したがN64に劣りbenchmark-only、N=32 projectionとvocabularyはrow8維持。K二重bufferはN=1,024で改善した一方、
  wide約4%、K=2,048/N=2,048約8%のoperator回帰に対してfull-model利得が約0.4%だったため棄却した。PMCsはrocprofv3が0を返したため
  counter由来の性能claimをせず、ISA/resource/kernel timing/profileを採否根拠とした。
- 証拠の追跡済み要約は
  [`phase63-gfx1201-mxfp8-wmma-prefill-v2.json`](../../../../../../ci/matrix/phase63-gfx1201-mxfp8-wmma-prefill-v2.json)を正本とする。

## 対象外

- SGLang、vLLM、その他非llama推論engineからのsource／疑似コード／tile table／symbolのcopy、adapt、port。
- MXFP8／MXFP6の量子化recipe、GGUF encoding、weight/activation default、standard MXFP8 E4 KV default、FP16 rollback、
  block16経路廃止の変更。
- persistent BF16/FP32 weight expansion、FP32 attention/KV保存、silent whole-layer dequant fallback。
- exact `gfx1030`／`gfx942`の新matrix kernel、MXFP6／NVFP4のproduction provider完成、MoE grouped GEMM、multi-GPU、
  continuous batching、HIP Graphの全面導入。
- 1,000 tok/sを理由にcorrectness、数値分類、fallback、resource、cleanup条件を緩めること。

## 完了時に残す記録

- clean-room入力、独立実装根拠、provider／selector／workspace contract。
- baseline／candidateの式、演算順、accumulator、丸めstage、ISA、数値分類、oracle、quality差。
- operator/full-modelの全代表点、median/MAD、TTFT、GPU span、kernel構成、VRAM、dispatch、fallback、cleanup。
- shared／scoped／benchmark-only／rejectedの採否、adoption scope、rollback provider、未達時の主残差と再検討条件。

[全体計画](../../../../main-plan.md) /
[Phase 37以降のロードマップ](../../../../active/2026/08/21-31/phase37-plus-mi300x-and-llama-gap-roadmap.md) /
[対応する履歴](../../../../../history/2026/09/1-10/phase63-gfx1201-mxfp-matrix-prefill.md)
