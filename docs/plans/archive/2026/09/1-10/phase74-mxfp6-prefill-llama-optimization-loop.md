# Phase 74: MXFP6 prefill llama.cpp比較最適化ループ

状態: `完了（2026-09-03）`

## 完了要約

比較→実装→benchmarkを3回実施した。Loop 1ではexact `gfx1030`へID47 half2 dot2 32x32を採用し、
4B 3 warmup＋10 measuredの512／2,048入力を旧ID25比`1.636倍／1.601倍`へ改善した。Loop 2ではexact
`gfx1201`へID48 packed E3M2x4→E4M3x4 SWAR ingressを採用し、旧ID45比`1.315倍／1.192倍`へ改善した。
Loop 3のactivation quantizer packed-store候補は両targetで約0.35〜0.84%退行したため棄却し、製品sourceから除去した。

最終sourceの既定経路は4B 1 warmup＋3 measuredで`gfx1030=411.82／393.60 tok/s`、
`gfx1201=2,843.97／2,959.30 tok/s`を再現した。27B 512入力も`57.22／475.51 tok/s`で両target PASSし、
生成token一致、HIP-only、fallbackなし、cleanup正常、resident `24,115,002,880` byte、peak
`24,776,887,808` byteを確認した。詳細なidentity、correctness、残差profile、llama.cpp比較は対応履歴と追跡要約を正本とする。

## 目的

Qwen3.5 denseのOCP MXFP6 E3M2 W6A6 prefillに限定し、固定llama.cpp Q6_Kとの実装・profile比較から
次の候補を一つ選び、実装、operator検証、full-model benchmark、採否、残差の再比較までを一つのループとして反復する。
同じ6-bitでも数値形式が異なるためQ6_Kを性能oracleやsLLMの新形式にはせず、GPU別tile、packed load、dot primitive、
MMQ構成、shape selectorの成熟した実装例として扱う。

今回の比較で確認した最大の残差は、exact `gfx1030`で現行ID25 tiled16がpacked E3M2をscalar展開していることと、
exact `gfx1201`のID45にもE3M2からE4M3へのtile ingress、LDS staging、scale処理が残ることである。これらを
persistent BF16／E4M3 weight展開やFP32 attention／KVへ置き換えず、packed resident MXFP6のまま改善する。

## 固定スコープ

- sLLM: OCP MXFP6 E3M2、block 32、E8M0 scale、W6A6、FP32 accumulation、BF16 RNE output。
- 比較対象: fixed `llama.cpp` commit `3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`のQ6_K HIP経路。
- GPU: Radeon Pro V620 exact `gfx1030`とRadeon AI PRO R9700 exact `gfx1201`。target別candidateとselectorを許容し、
  一方のGPUで不採用でも他方の採用を妨げない。
- 主model: fixed Qwen3.5-4B dense。採用候補のmodel非依存性確認だけにQwen3.5-27B denseを使う。
- workload: direct pretokenized input 512／2,048、明示FP16 KV、単一GPU、greedy、ignore EOS、短いoutput budget。
  採否指標はprefill time／prefill token/s、kernel duration、peak／resident VRAMであり、decode token/sは扱わない。
- Phase開始時にsLLM source/build、ROCm、両GPU UUID/BDF、model lock／GGUF、llama.cpp build／GGUF、command、
  input token列を固定する。Phase計画時のsLLM sourceは
  `586c27b60d976781963b4a3e0901f9be3cb2c9e2`だが、実行開始時のclean candidateを証拠identityとする。

今回の1 warmup＋3 measuredによる探索値は次のとおりであり、最終採否用baselineは同一の固定identityで取り直す。

| target | sLLM MXFP6 4B prefill | llama.cpp Q6_K 4B prefill | 現在の差 |
| --- | ---: | ---: | ---: |
| `gfx1030` | 253.28 tok/s | 1,967.06 tok/s | llama.cpp 7.77倍 |
| `gfx1201` | 2,170.05 tok/s | 3,598.16 tok/s | llama.cpp 1.66倍 |

## 非目標

- `M=1` decode kernel、decode token/s、sampling、KV更新、decode graph／fusionの改善。
- MXFP8、NVFP4、MXFP4、BF16、MoEへの移植。共通化可能な成果は後続候補として記録するだけで自動実装しない。
- Q6_Kの読込み・生成・公開、一般的なllama.cpp INT量子化形式の製品対応。
- MXFP6のblock、scale、accumulation、output、特殊値規則の変更。
- persistent BF16／FP16／E4M3／FP32 weight、FP32 attention／KV、モデル全体の展開cache。
- 新GPU、新model architecture、複数GPU、continuous batching、WebUI／server経路。
- llama.cppとの完全同等をPhase完了条件にすること。比較差は候補順位と残差判断に使う。

## 受入条件

1. 両targetでcurrent sLLMとfixed llama.cppの512／2,048 prefill baselineを取得し、format差とtiming境界を明記する。
2. sLLMのactivation quantization、MXFP6 matmul shape／kernel別duration、attention、GDN、otherを分離し、最初の候補を
   fresh profileの寄与順から選ぶ。無効なcounterを帯域・occupancyの根拠に使わない。
3. llama.cpp Q6_KのRDNA2／RDNA4 MMQ、MMVQ、packed tile load、Q8_1 activation、integer dot、shape別構成を比較し、
   sLLMへ適用可能な機構と、Q6_K固有で適用不能な機構を分けた比較記録を残す。
4. 候補は一度に一つだけ実装し、独立FP32 oracle、特殊値、repeat、非整列値とselector境界をPASSしてから
   full-model性能を測る。correctness／security defect、fallback、非HIP dispatch、cleanup failureは採用不可とする。
5. 採用はtarget単位で判断する。同一固定binaryのpaired測定で対象prefill行が安定して改善し、同じtargetの他方の入力長や
   scope外selectorを実質退行させない候補をscoped defaultにできる。固定の必達倍率は設けず、差が測定揺らぎ以下なら棄却する。
6. 採用または棄却後に残差を再profileし、llama.cppとの差分表を更新して次候補へ戻る。候補が二回連続で棄却された場合は
   同じwork unitを細分化せず、残差と仮説を再順位付けする。
7. 最終採用候補だけをQwen3.5-27BのVRAMに収まる512入力で両target確認し、model名に依存しないshape selector、
   prefill速度、生成token、HIP-only、fallback、resident／peak、cleanupを記録する。
8. 採用source、棄却candidate、残存するllama.cpp差、転用可能な共通primitiveをhistoryと追跡要約へ同期する。

## 比較→実装→benchmarkループ

### P74-A: 比較とbaseline

1. sLLMとllama.cppの固定identityで4B 512／2,048を取得する。
2. 両engineのprefill matrix providerを、weight／activation layout、tile、load再利用、dot命令、scale適用、同期、
   launch、shape selectorの軸で比較する。
3. sLLMをprofileし、targetごとに改善可能時間の大きい候補を一つ選ぶ。

llama.cpp sourceを直接再利用する場合はMIT provenanceをfile／範囲単位で記録する。構造だけを参考に独自実装する場合も、
比較記録に参照箇所と採用しなかったQ6_K固有契約を残す。

### P74-B: 単一候補の実装とoperator検証

- 最初のgfx1030候補は、ID25のscalar E3M2展開を減らすpacked group load、複数N列／M行再利用、integer dotまたは
  同等のformat-native accumulation構成を比較対象とする。E3M2／E8M0の実数式を変えるcandidateは別の数値変更として扱い、
  暗黙に採用しない。
- 最初のgfx1201候補は、ID45の4-value ingress、E3M2→E4M3変換、LDS配置、N64／N128 tile、scale適用位置を比較し、
  profile上の最大残差だけを変更する。
- activation quantizationの融合やmaterialization削減は、fresh profileで削減可能量が上位になった場合だけ候補化する。

operator matrixは少なくともM=`17/127/128/129/511/512/513/2048`、Kの32整列内外、主要projection、N=`1024/2560/9216/17408/32768`
とtail／上限外を含める。all 64 E3M2 code、E8M0 finite edge／NaN、signed zero、非有限値位置、BF16 RNE、repeatを確認する。

### P74-C: 段階benchmarkと採否

1. production shapeのoperator microbenchmarkで明確に遅い候補を早期棄却する。
2. 1 warmup＋3 measuredの4B 512／2,048でscreeningする。
3. surviving candidateだけを3 warmup＋10 measuredでcurrent defaultとpaired比較し、中央値、MAD、prefill time、
   token/s、dispatch、resource、VRAMを保存する。
4. 採否後のproduction defaultをprofileし、P74-Aの差分表へ戻る。

### P74-D: 最終転用確認と完了

両targetについて採用された最終candidateを同一最終sourceからbuildし直し、4Bの最終比較と27B 512転用確認を行う。
一方のtargetで候補なしの場合は現行providerを明示維持する。llama.cpp未達や候補なしを失敗へ読み替えず、試した候補と
残差が再現可能ならPhaseを完了できる。

## 初期候補順位

1. `gfx1030`: ID25 tiled16のscalar unpack／FP32 FMAを置換するpacked multi-output MMQ。
2. `gfx1201`: ID45のE3M2→E4M3 ingressとLDS staging削減、shape別N tile再評価。
3. 両target: activation quantizationの再利用／融合。ただし既存約5%寄与を上限と決めつけずfresh profileで再確認する。
4. matrix残差縮小後のattention／GDNは計測結果へ記録するが、このPhaseでは実装しない。

この順位はAI作成時の非blocking仮説であり、対象はPhase 74、費用は各targetのprofile一式と候補ごとのoperator／4B測定、
有効期限はP74-Aのfresh profile確定までとする。P74-Aの寄与順が異なれば、ユーザー確認を待たずPhase内で順位だけを更新できる。

## 停止・再計画条件

- correctness/security、数値契約、selector fail-closedを満たさないcandidateは性能測定へ進めず棄却する。
- 同じwork unitが二回連続で棄却、検証が実装時間の30%を超過、または比較・実装・測定の一巡で新しい削減可能残差が
  得られない場合は、そのwork unitを停止して残差順位を更新する。
- profileでMXFP6 matmul以外が支配的になってもdecode、attention、KV、他formatへscopeを拡張せず、Phase 74を完了して
  後続候補としてmain planへ返す。

[全体計画](../../../../main-plan.md) /
[Phase 37以降のロードマップ](../../../../active/2026/08/21-31/phase37-plus-mi300x-and-llama-gap-roadmap.md) /
[履歴](../../../../../history/2026/09/1-10/phase74-mxfp6-prefill-llama-optimization-loop.md) /
[追跡要約](../../../../../../ci/matrix/phase74-mxfp6-prefill-llama-optimization-loop-v1.json)
