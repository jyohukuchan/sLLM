# Phase 37以降: 性能・機能ロードマップ

## 目的

2026-08-22のユーザー指示により、直近の性能最適化はV620 exact `gfx1030`だけで開始した。
2026-08-23のユーザー指示により、Phase 49は全7比較行の同等達成を後続GPUの開始条件にせず、
GQA P32を限定採用、long-prefill v2とHIP Graphを棄却し、採用経路の退行を確認して完了した。
Phase 50はR9700 exact `gfx1201`の限定採用とMI300X exact `gfx942` wave64引継ぎ準備を完了し、
実機性能検証をPhase 51へ引き継いだ。2026-08-25のユーザー指示でPhase 51を再開して完了し、Phase 52はR9700
`100,000/2`に残ったKV物理コミットOOMを解消して完了した。
番号上の既定順は49→50→51→52だが、Phase 51と52は独立に開始でき、V620またはR9700の同等達成を後続GPU開始の必須条件にはしない。

2026-08-21に作成した旧Phase 37〜38のMI300X先行計画はコード変更・GPU実行前に再編し、その作業範囲を
Phase 51へ吸収する。Phase 39〜45は完了済みである。2026-08-26のユーザー指示によりPhase 46を次の機能経路として具体化し、
16値scaleのKV FP8形式実装とtarget別default採用をPhase 53へ割り当てる。Phase 47〜48は予約済みのまま保持する。
Phase 49〜53の性能・KV経路とPhase 46のtool／品質評価基盤は完了済みである。2026-08-27のユーザー指示により、
block16の精度改善研究を独立したPhase 54へ割り当てる。
default判定だけがPhase 46のfreeze済みKV品質policyに依存する。
2026-08-30のユーザー決定でPhase 53/54のblock16製品経路を廃止し、reviewed Qwen3.5-4B BF16 dense textでは
standard OCP `kv-mxfp8-e4`（E4M3FN、block 32、E8M0 scale）を省略時KVへ採用した。明示`fp16`はrollbackとして残す。
block16のschema、policy、research source、証拠は互換入力やproduction候補ではなく、監査可能な履歴としてだけ保持する。
Phase 55〜62のモデル機能・MXFP W/A／共通codec統合は完了済みである。2026-08-31のユーザー指示により、Phase 61に残った
MXFP性能残差を、matmul固有の追加最適化ではなく、KV append／attention／NVFP4等にも再利用できるlow-precision primitiveへ
分離して改善する作業をPhase 62へ割り当て、両RDNAのbit-exact検証と性能採否まで完了した。
Phase 63〜65のexact gfx1201 MXFP8 WMMA最適化も完了した。2026-09-01のユーザー指示により、Phase 65後のlinear／attention残差を
MXFP8で共通providerとして実証し、MXFP6、NVFP4／MXFP4、BF16 attentionへ実候補を移植するPhase 66も完了した。
Phase 67〜69ではexact gfx1030のMXFP8 software-MMQを段階的に改善し、ID41 vector32 ingressまで限定採用した。
2026-09-02のユーザー指示によるPhase 70では、両RDNAのMXFP6をpacked residentのままMXFP8 MMQ／WMMA骨格へ接続した。
追加P70-Fでgfx1201 ID45 packed-group N64をshape限定採用し、ID46 N128とgfx1030 ID43はbenchmark-onlyとして完了した。

この計画はユーザー指示によりPhase番号と順序を割り当てる。Phase 36以前の完了条件を遡及変更せず、
角括弧で将来項目だったResponses APIとWebUIも後続Phaseへ割り当てる。各Phaseのcorrectness/security条件は必須とする。
性能値は採用判断と再計画に用いる目標であり、数値未達を隠すために比較条件やモデルを変更しない。

## 正本と基準値

- 製品要件: repository外の`sLLM.md`。
- 全体計画: [main plan](../../../../main-plan.md)。
- API: [OpenAI compatibility profile](../../../../../api/openai-compatibility.md)。ResponsesやAnthropic Messagesは、
  実装前に別のversioned profileと外部仕様pinを追加する。
- runtime: [runtime architecture](../../../../../architecture/runtime.md)。transportからgeneration、token selection、
  model state、HIP providerを分離する。
- model identity: [model lock](../../../../../models/model-lock.md)。cache、adapter、checkpoint、dynamic model lifecycleで
  verified identityを弱めない。
- V620の計画開始値は[Phase 35 archive](../../../../archive/2026/08/11-20/phase35-long-context-full-attention-gdn-optimization.md)の
  10,001/2 sLLM `22.683`秒を参考値とする。固定llama.cppを含む通常5行＋長時間2行の基準値はPhase 49開始時の同一source・同一測定で
  取り直し、歴史値を新しい候補の比較値に流用しない。
- R9700の直近比較は[R9700 E2E history](../../../../../history/2026/08/21-31/r9700-sllm-llama-e2e-comparison.md)の
  10,001/2 sLLM `3.936429665`秒、固定llama.cpp `2.063845785`秒、比`1.90733x`である。これはPhase 50の
  参考値であり、7行のPhase 50基準値を置き換えない。
- MI300Xの参考値は[Phase 36 archive](../../../../archive/2026/08/11-20/phase36-mi300x-current-main-validation.md)と
  [Session D summary](../../../../../../ci/matrix/phase36-mi300x-session-d-summary-v1.json)。Qwen3.5-4B BF16、FP16 KV、
  input ID `23066`を10,001個、greedy 2 output、3 warmup＋10 measuredで、sLLM E2E中央値は
  `22.556130816`秒、fixed llama.cppは`0.8512540725`秒、E1比は`26.4975x`だった。
- Session Dのrocprofv3 GPU時間比はGDN `73.95%`、Full Attention `25.12%`、projection `0.70%`、other
  `0.23%`。Phase 51では新しい7行profileを優先し、この値は候補選択の参考にだけ使う。
- Phase 36のVMは削除済みである。Phase 51で新しいexact gfx942実機を確保するまではcompile、selector、host oracle準備だけを
  draftとして進め、compile成功や過去のSession D証拠を新しい候補のGPU PASSへ読み替えない。
- 比較は同じupstream revision、**input** token列、dtype、KV、GPU、warmup/反復、timing boundaryを固定する。
  GGUF bytes/tensor setが異なる間はE1 system-equivalentとし、strict-identicalと表記しない。生成／visible token列とstop reasonは
  各engine内の反復一致をhardに確認し、cross-engine差はbounded digest観測として扱う。

## 全体順序

| Phase | 状態 | 主範囲 | 主要依存 |
| --- | --- | --- | --- |
| 37 | replanned-before-start | 旧gfx942 GDN・Full Attention計画をPhase 51へ吸収 | production source・GPU証拠の変更なし |
| 38 | replanned-before-start | 旧MI300X残差計画をPhase 51へ吸収 | production source・GPU証拠の変更なし |
| 39 | complete | service operability・認証・observability基盤 | 現行profile v1 |
| 40 | complete | sampler chain、GPU sampling、logprobs、grammar/structured generation | 現行generation loop |
| 41 | complete | prefix/KV reuse、session checkpoint、context shift、speculation | opaque KVとPhase 40 token selection |
| 42 | complete | Completions・Embeddings・Rerank・token utility・infill endpoint | transport-independent modes |
| 43 | complete | Responses・Anthropic Messages・function/tool protocol | Phase 40・42 |
| 44 | complete | generic template、reasoning control、interactive CLI | Phase 41・43 message/state。MI300X実機はdeferred |
| 45 | complete (host + RDNA GPU; MI300X deferred) | LoRA/control vector、dynamic model lifecycle/router cache | model lock・Phase 39 ops |
| 46 | complete | conversion、quantization、benchmark、quality/debug tools | stable GGUF/model identities、gfx1030 FP16 baseline、freeze済みKV policy |
| 47 | approval-required | 組込みtool/MCP実行 | Phase 39 security・Phase 43 tool protocol |
| 48 | prototype-complete・継続中 | minimal WebUI/server UI | Phase 39・42〜45 public APIs |
| 49 | complete-scoped-adoption | V620のGQA P32を限定採用、long-prefill v2とHIP Graphを棄却 | 3候補の関連実機行、通常5行退行確認、採否履歴 |
| 50 | complete-limited-adoption | R9700 `gfx1201`採用とMI300X `gfx942` wave64引継ぎ準備 | Phase 49完了（充足済み） |
| 51 | completed-target-separated | Phase 49/50採用内容のMI300X wave64適用・実機検証 | MI300X 7/7 PASS、GDN候補を既定無効でtarget分離 |
| 52 | complete | R9700 `100,000/2`のKV物理コミットOOM解消 | 自動経路4/4 PASS、10,001 regression 13/13 PASS |
| 53 | complete-superseded | block16 descriptor v2のtarget別評価を完了。2026-08-30の経路廃止により採用候補ではなく履歴化 | Phase 46のfreeze済みpolicyによる当時の評価を保持 |
| 54 | complete-route-retired | block16精度研究は勝者なしで完了し、製品経路を廃止。standard OCP MXFP8 E4をreviewed scopeの既定KVへ採用 | explicit FP16 rollbackを維持、gfx942実機は一括検証へ延期 |
| 55 | complete | Gemma 4 26B-A4B MoEのNVFP4 CLI/API/WebUI production統合 | 既存Gemma/Qwen MoE、NVFP4、dynamic model lifecycle |
| 56 | complete | Gemma 4 12B公式assistantによるMTP統合 | Phase 55 lifecycle、逐次target同値contract |
| 57 | complete-foundation | DeepSeek V4 Flash identity／semantic／capacity foundation | model lock、GGUF、専用route operator |
| 58 | complete-foundation | MiniMax M3 identity／MSA／MoE／capacity foundation | model lock、GGUF、専用route operator |
| 59 | complete-foundation | DiffusionGemma identity／semantic／capacity foundation | Gemma MoE backbone、GGUF parser |
| 60 | complete | Ministral 3 3B text productionとRoPE修正 | canonical GGUF、両RDNA、固定llama.cpp |
| 61 | complete-model-evaluated | OCP MXFP8／MXFP6 W/A統合と両RDNA operator、gfx1030 full-model評価 | OCP MX v1.0、GGUF recipe、Qwen dense lowering |
| 62 | complete-shared | 再利用可能low-precision scalar/block codecとMXFP target別最適化 | 両RDNA bit exact、W/A・KV・attention改善、reuse/fusion棄却 |
| 63 | complete-scoped-adoption | exact gfx1201 large-M MXFP8 WMMA providerとprepared shape selector | 24 operator case、final ISA、品質、3+10 full-model、profile |
| 64 | complete-scoped-adoption | exact gfx1201 MXFP8のweight direct-loadとshape selector | 4B／9B full-model、operator、profile、資源 |
| 65 | complete-scoped-adoption | exact gfx1201 MXFP8のactivation／weight direct-load | model非依存ID36、4B／9B 2,048-token、profile |
| 66 | complete-scoped-adoption | prepared low-precision provider、MXFP8 ID37限定採用、typed attention棄却、MXFP6／NVFP4／MXFP4・BF16 attention移植 | exact gfx1201 operator、4B／9B full-model、reviewed model route |
| 67 | complete-scoped-adoption | exact gfx1030 MXFP8のN方向再利用とID27 col8 selector確定 | 18-case operator、4B 512／2,048、profile |
| 68 | complete-internal-fast-path | exact gfx1030 MXFP8のE4／scale分離と内部MX normal fast path | E4分布、特殊値、4B paired full-model |
| 69 | complete-scoped-adoption | exact gfx1030 MXFP8のvectorized E4 ingress ID41 | 28-case operator、resource／counter、4B 3+10 full-model |
| 70 | complete-gfx1201-scoped-adoption | exact gfx1030／gfx1201 MXFP6のMXFP8 MMQ／WMMA骨格再利用 | gfx1201 ID45限定採用、ID46／gfx1030 ID43 benchmark-only、両target operator／4B full-model |

直近の性能laneの番号上の既定順はPhase 49→50→51→52である。Phase 49の3候補判定と採用経路の退行確認、Phase 50のR9700採否と
MI300X wave64引継ぎ準備は完了した。Phase 51は一時保留中にR9700限定のPhase 52を先に完了し、2026-08-25のユーザー指示で再開して完了した。Phase 49では候補routeをexact `gfx1030`へ、Phase 50では
採用routeをexact `gfx1201`へ限定した。Phase 50の全7行llama.cpp同等達成は後続Phase開始のgateではない。Phase 46も完了し、
Phase 53のdescriptor v1／v2評価とPhase 54の改善研究は当時の証拠として完了済みである。2026-08-30のユーザー決定により
block16製品経路は廃止し、両local targetを含むreviewed Qwen3.5-4B BF16 dense textの省略時KVをstandard OCP
`kv-mxfp8-e4`へ変更した。gfx942のfresh実機証拠は追加のMI300X検証項目がまとまった時点の一括実行候補へ延期する。
Phase 55〜70は完了済みである。次のlow-precision候補は独立NVFP4 specializationで、Phase番号は未割当である。
Phase 47〜48は内容と番号を保持する。

複数surfaceへ現れる機能の所有権は一つに固定する。Phase 39はresumable transport/replay、Phase 40はsamplerと`n` choice
state、Phase 41はassistant-prefill/state semantics、Phase 42はFIM/infill execution modeを所有する。Phase 42〜44の後続記述は、
所有済み機能を各wire profile、renderer、CLI/UIへ接続するadapter範囲であり、別実装や別state machineを作らない。

## 共通実施規則

1. 各Phase開始時に、対象surface、非目標、仕様pin、受入case、source/build/model identityを固定する。
2. llama.cppから直接reuseする場合はMIT provenanceをfile単位で記録する。llama.cpp以外はno-copy referenceを維持する。
3. host unitは非整列値と境界の両側を含める。GPU PASSはexact target、数値oracle、HIP dispatch、fallbackなし、
   cleanupを必要とし、timeout、crash、0 caseをPASSにしない。
4. draftはaffected test、integrationは影響matrixと一回のintegration review、release/pushはclean candidateの最終gateを使う。
   各checkpointのfresh reviewや全GPU rerunを要求しない。
5. public APIは各Phaseの最初に外部schema/profile pin、rejection matrix、security/provenance境界を固定する。
   未知fieldを黙って無視せず、versioned schemaに従い4xxでfail closedにする。prompt、token、secretをmetric/logへ出さない。
6. 性能candidateは同一sourceでbaseline/candidateをcounterbalanceし、median、MAD、全反復値、kernel family、VRAM、
   process/healthを記録する。局所改善だけでwall改善を主張しない。
7. 既存gfx1030/gfx1201 routeを変更する共通sourceは該当targetのfocused regressionを行う。gfx942固有sourceだけなら
   RDNA GPU rerunを常に要求せず、compile/dispatch selector testで非選択を証明する。

## Phase 37〜38: 実装前の再編

- 旧Phase 37のgfx942 GDN／Full Attentionと、旧Phase 38のMI300X残差解消は、production sourceやGPU証拠を
  変更する前に中止し、Phase 51へ吸収した。完了または棄却した性能候補としては扱わない。
- 旧計画で固定したwave64、FNUZ、4 KV encoding、`contiguous-resident`、数値変更分類、VM実測のfail-closed条件は
  Phase 51へ引き継ぐ。Phase 36の履歴と証拠は変更しない。

## Phase 49〜51の共通性能契約

### 比較対象

- モデルはQwen3.5-4Bの固定revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`、BF16 weight、
  FP16 KV、MTPなし、visionなし、greedyとする。sLLMとllama.cppのGGUFが異なる間はE1 system-equivalentと表記し、
  strict-identicalとは表記しない。
- 比較対象は固定llama.cpp `b10453` / `3cb7ffb1a1f612d5e4a46244ae5a3c77ad934a70`とする。
  Phase開始時に両engineのsource、binary、model、runner、ROCm、GPU identityを固定する。
- ここで「batchingなし」は要求batchと並行sequenceを1に固定する意味である。両engineとも単一GPU、active request 1、
  parallel slot 1、要求の重なり0とし、continuous request batching、複数GPU、MTPを無効にする。
- llama.cppの`--batch-size`やsLLMのchunked prefillのような、単一prompt内部の処理単位を1へ強制する意味ではない。
  内部chunk／tileは各engineの実運用既定値を使い、測定前に固定・記録する。設定context長は両engineで揃える。

### 固定比較matrix

通常matrixはPhase 36 Session Dと同じinput token列／digestを再利用する5行とし、候補ごとの絞り込んだ性能回帰に使う。

| 種別 | 行 | input token数 | output token数 | 主に見る領域 |
| --- | --- | ---: | ---: | --- |
| 通常 | `short-odd` | 17 | 17 | 短いprefillとdecode、非整列値 |
| 通常 | `32-32` | 32 | 32 | 短い均衡形状 |
| 通常 | `prefill-long` | 1,024 | 128 | prefill寄り |
| 通常 | `decode-long` | 32 | 256 | decode寄り |
| 通常 | `long-10001` | 10,001 | 2 | 長いcontextのprefill／attention |

長時間matrixは次の2行とする。実行時間が長いため候補ごとの通常回帰には含めず、Phase開始時の基準値、
該当経路へ影響する候補群の採否前、Phase最終候補で実行する。

| 種別 | 行 | input token数 | output token数 | 主に見る領域 |
| --- | --- | ---: | ---: | --- |
| 長時間 | `long-100000` | 100,000 | 2 | 100k contextのprefill、attention、KV、memory |
| 長時間 | `decode-20000` | 32 | 20,000 | 長時間decode、状態更新、同期、持続性能 |

`long-100000`はtoken ID `23066`を100,000個使う。`decode-20000`は`decode-long`と同じ32-token入力を使い、
両engineでEOSと追加stopを無効化してmax output 20,000まで必ず実行する。長時間2行は両engineとも設定context長を
`131,072`に固定する。実行前にVRAM見積りと空き容量を確認するが、OOMやtimeoutを行の省略やPASSへ読み替えない。

単一行の改善、10,001/2だけのPASS、通常5行だけのPASSを「llama.cpp同等」と呼ばない。全7行で固定input token列、
output budget、protocolを一致させる。各engine内の全warmup／measured反復では生成token列、visible token列、stop reasonの
一致をhardに確認する。異なるGGUF tensor set／converterを使うE1 system-equivalent比較では、cross-engineの生成／visible／stop
一致はdigestとboolで観測し、性能gateの阻害条件にはしない。cross-engineのinput列、output budget、protocol、各engine内の
決定性・shape・stop形式・HIP／resource／cleanup契約は引き続きhardとする。

### 測定と同等判定

1. 通常5行は各engineを3回warmup後に10回測定する。長時間2行は費用を抑えるため1回warmup後に3回測定し、
   通常行より確度が低いことを要約へ明記する。いずれもengine順をcounterbalanceし、進行中のprocess、GPU health、
   token進捗を監視する。進行している長時間runを任意の短いcheckpoint時間だけで終了しない。
2. E2E、TTFT、prefill、TPOT、token/s、peak VRAM、GPU family時間、全反復値、median、MADを記録する。
   長時間行のraw profilerは常時取得せず、候補群の分解または最終代表runだけで取得する。
3. 全7行のE2Eで`sLLM median <= llama.cpp median + max(sLLM MAD, llama.cpp MAD)`を満たすことを
   「測定上遅くない」と定義する。さらに全7行のTTFTと、output 17以上の5行のTPOTも同じ条件を満たす。
4. 比率`median_sLLM / median_llama`を併記する。MAD幅を利用して明白な悪化を隠さず、外れ値除去、測定後の行変更、
   測定後のchunk/context設定変更、失敗runの除外を行わない。
5. 全行でHIP-only、fallbackなし、partial offloadなし、非finiteなし、cleanup 0、実行前後のhealthと資源復帰を確認する。
6. Phase 49〜51では同じ7行と式で同等達成の有無を報告する。ただしPhase 49は3候補の採否と退行確認で完了でき、
   全7行同等達成そのものを後続targetの開始条件にしない。Phase 50〜51でも同等達成を相互の開始条件に追加しない。

## Phase 49: V620限定のllama.cpp同等化

> 状態: complete-scoped-adoption（2026-08-23）。全7行同等達成ではなく、3候補の採否と通常5行退行確認で完了。

### 範囲と作業単位

1. exact `gfx1030`のV620 1台だけで通常5行＋長時間2行のsLLM／llama.cpp基準値とGPU profileを取り直す。R9700とMI300Xは
   性能候補の実装・採否・GPU回帰へ使わない。
2. 各行のE2E残差をGPU family、host wait、H2D/D2H、provider、shapeへ分解し、Amdahl上限の大きい順に一つずつ扱う。
   Phase 35後に残るFull Attention、projection、実行時dispatch／同期、decode、loaderを候補一覧とするが、fresh profileで
   上位でない候補は実装しない。
3. 新providerは最初にexact `gfx1030`だけへrouteし、gfx1201/gfx942は既存baselineを維持する。共通algorithmを使える設計でも、
   他targetのselectorをPhase 49で有効化しない。
4. operator oracleは`1/3/17`、tile境界の`B-1/B/B+1`、tail、非整列shape、NaN/Inf、state/KV transactionを含める。
   candidateごとに数値、資源、局所時間、通常5行を確認し、採否理由を記録する。長時間2行は個別candidateごとに実行せず、
   100k prefill／KVまたは20k decodeへ影響する候補をまとめた採否時点で実行する。
5. long-prefill v2、GQA P32、HIP Graphを現在の最終候補とし、それぞれ関連する実機行で採用または棄却する。
   判定後は採用候補を含む通常5行で重大な退行がないことを確認し、残るllama.cpp差はPhase 50以降へ持ち越す。
6. V620実行前にローカルQwen補助serviceを停止してV620 2台を解放する。性能測定はcanonical V620 1台だけを使い、
   spare V620をtensor parallelや要求batchへ使わない。この期間の補助作業はnative Codex subagentを使う。

### 完了条件

- long-prefill v2、GQA P32、HIP Graphの3候補について、関連する実機性能、数値、fallback、後始末、selectorを確認し、
  採用または棄却とその理由を固定する。性能未達や候補棄却は隠さないが、全7行同等達成は完了条件にしない。
- 採用候補を含むcurrent candidateの通常5行で正しさ・資源条件をPASSし、Phase 49開始時または直前採用候補からの
  原因不明な重大退行がないことを確認する。
- exact `gfx1030`以外へ新しい性能routeを開かず、R9700／MI300Xの既存routeを変更していないことをselector testで示す。
- 採用source、棄却candidate、取得済み7行測定、未達行と残差profileを履歴へ固定してからPhase 50／51を開始する。

### 完了結果

- GQA P32はexact `gfx1030`、decode、GQA4、head dimension 256、FP16 KV、KV長4,096以上へ限定して既定有効化した。
  `32/20,000`ではE2E中央値を`934.262`秒から`529.331`秒へ43.34%短縮し、同一20,000-token digest、HIP-only、
  fallbackなし、後始末0を確認した。
- long-prefill v2はoperatorで10k入力まで52.96〜58.60%短縮したが、`100,000/2` full-modelの単一warmupが約33分を要し、
  current controlの1 warmup＋3 measured合計より遅かったため不採用とした。実装は明示的opt-inへ隔離し、既定経路に入れない。
- HIP Graphは無効時の`17/17`がPASSした一方、有効時にSIGSEGVしたため不採用とし、候補固有APIと実装を撤去した。
- 最終通常5行は各3 warmup＋10 measuredで5/5 PASSした。E2E中央値は`17/17` 423.961 ms、`32/32` 750.651 ms、
  `1,024/128` 4,214.241 ms、`32/256` 5,779.410 ms、`10,001/2` 13,507.666 msで、Phase 49開始時比24.24〜45.43%短縮した。
  全行でexact `gfx1030`、HIP-only、fallbackなし、反復一致、要求後とprocess終了後の資源復帰を確認した。
- 固定llama.cpp比のE2E残差は順に+0.78%、+2.16%、+3.04%、+6.65%、-9.45%である。全7行同等とは主張せず、
  current controlの`100,000/2`約295.093秒対llama.cpp約194.121秒、P32採用後の`32/20,000`約529.331秒対
  llama.cpp約428.989秒をPhase 50以降へ持ち越す。

## Phase 50: R9700実機移植とMI300X wave64引継ぎ準備

> 状態: complete-limited-adoption（2026-08-24）。R9700 `gfx1201`の採否とMI300X `gfx942` wave64引継ぎ準備を完了。

### 範囲と作業単位

1. exact `gfx1201`、ROCm 7.14、Code Object V6、wave32のtarget専用成果物でR9700の通常5行＋長時間2行と
   固定llama.cpp baseline/profileを新規取得する。既存10,001/2比較は参考値だけとする。
2. Phase 49変更をtarget共通、gfx1201で再測定するwave32候補、gfx1030限定、不採用、gfx942 wave64再設計へ分類する。
   fresh profileの残差順にGQA split、decode融合、attention/linear、matmul、execution制御を個別採否し、全候補の実装を要求しない。
3. 共通source変更時はV620通常5行、長時間経路へ影響する場合だけ該当長時間行を再確認する。gfx1201固有変更なら
   gfx1030/gfx942非選択をselector testで示す。
4. MI300Xはexact `gfx942:sramecc+:xnack-`／wave64 compile、host selector非選択、意味・数値・ABI・workspace引継ぎまでを扱う。
   MI300X実機7行と性能採否はPhase 51に残し、compile成功をGPU PASSへ読み替えない。

### 完了条件

- R9700の7行で規定反復、正しさ、HIP-only、fallback、資源、固定llama.cpp差、未達理由を記録し、Phase 49変更を
  gfx1201採用・target分離・baseline/decompose・不採用・gfx942再設計のいずれかへ分類する。
- 共通source変更の影響範囲でV620 Phase 49 closeoutを維持し、gfx1030 P32経路を原因不明に退行させない。
- exact gfx942 compile/link、host selector非選択、wave64引継ぎ台帳をPhase 51の入力として固定する。
- R9700の全7行llama.cpp同等は目標と報告項目だが、Phase 50完了またはPhase 51開始のhard gateにしない。

### 完了結果

- R9700 exact `gfx1201`、Code Object V6、wave32の最終7行は6/7 PASS、1/7 FAILだった。PASS行は全てHIP-only、fallbackなし、
  反復一致、cleanup 0である。E2E中央値（sLLM／固定llama.cpp、ms）は、`17/17` `407.915/332.726`、
  `32/32` `759.729/604.069`、`1,024/128` `3,383.627/2,509.156`、`32/256` `5,959.860/4,712.364`、
  `10,001/2` `4,002.834/2,072.476`、`32/20,000` `532,486.026/377,632.768`だった。
  `100,000/2`はlayer 31のKV commitでOOMとなり、未達理由を記録した。最終比較と失敗を含む追跡済み要約は
  [`ci/matrix/phase50-r9700-summary-v1.json`](../../../../../../ci/matrix/phase50-r9700-summary-v1.json)に固定した。
- Phase 49変更は、target共通意味契約、exact `gfx1201`での residual RMSNorm、GDN projection bundle、MLP gate-up-SiLU bundle、
  GQA4 P32（KV長4,096以上）を採用し、`gfx1030`限定経路、不採用経路、gfx942 wave64再設計へ分類した。llama.cpp同等未達は
  残差として報告するが、完了条件にはしなかった。
- 共通source変更後のV620 exact `gfx1030`通常5行は5/5 PASSで、Phase 49 closeout比は`-0.21〜+1.16%`に収まった。
- exact `gfx942`のCargo build、feature compile/link probe、host selector非選択はPASSした。MI300X実機の7行性能検証と採否、
  `project-verified`昇格は未実施であり、Phase 51が所有する。wave64ではwave32のlane ownership、block、LDS/register、barrier、
  GQA partitionを直接流用せず再設計する。
- Phase 50後の`100,000/2` OOM分析で、16 GiB超を一律16K候補から評価する自動prefill selectorを修正した。capacity tierは
  24 GiB未満512、24〜35 GiB未満2K、35〜60 GiB未満4K、60〜160 GiB未満8K、160 GiB以上16Kであり、各tierの下位候補は
  exact graph memory見積りで選ぶ。32 GiBのV620/R9700は2K開始へ変わるため、既存Phase 49/50測定は履歴として維持し、
  current candidateの性能主張には再測定を要する。修正後のR9700 `100,000/2`自動再実行ではHBM peakが
  `26,414,587,904` bytesから`13,160,554,496` bytesへ約50.18%低下したが、約`152.867`秒後、layer 23の
  virtual KV physical commitmentで再びOOMとなった。実効2K／512を失敗rowへ保存できていない問題を含め、Phase 52で解消する。
  この残件はPhase 51の開始gateにはしない。

実行tuple、7行、selector境界、candidate順、停止／再計画条件、証拠は
[Phase 50詳細計画](../../../../archive/2026/08/21-31/phase50-r9700-port-and-mi300x-handoff.md)を正本とする。

## Phase 51: MI300Xへの適用と検証

> 状態: completed-target-separated（2026-08-25）

### 範囲と作業単位

1. Hot AisleのMI300X VF x1、exact `gfx942:sramecc+:xnack-`、wave64、ROCm 7.14 tupleを再確保し、
   Phase 49完了candidateと、利用可能ならPhase 50の採用内容を含むcurrent candidateで通常5行＋長時間2行と新しいGPU profileを取得する。
   単一VMの結果を別CDNA SKUへ一般化しない。
2. Phase 49〜50のalgorithm／layout／同期削減をwave64へ移植する。wave32 kernel binaryや閾値をそのまま選択せず、
   head／state column ownership、tile、reduction、barrier、LDS/registerをgfx942向けに再決定する。
3. 旧Phase 37〜38のGDN column-state、tiled Full Attention、wave64 MMVF、FNUZ hipBLASLt solution、activation量子化共有、
   command-list／graph replay、KV provider候補を、新しい7行profileのAmdahl上限順に評価する。
4. FP16／FP8 accumulatorやsoftmax順序を変える候補は数値台帳のN0〜N3へ分類し、N2以上を性能だけで自動採用しない。
5. 共通sourceを変更した場合はV620とR9700の通常5行をcurrent candidateで再実行し、長時間経路へ影響する場合と
   Phase 51最終候補では長時間2行も再実行する。gfx942固有sourceだけなら、
   両RDNA targetの非選択をselector testで示し、不要なGPU再実行は要求しない。

### 完了条件

- MI300Xの通常5行と長時間2行で正しさ・資源条件をPASSし、移植候補を採用・target分離・不採用のいずれかへ分類する。
- llama.cpp同等条件の達成有無と残差を記録する。current candidateでV620のPhase 49 closeout状態を維持し、
  Phase 50で確立したR9700経路がある場合はその正しさと性能を退行させない。
- 全3 targetの最終7行比較、GPU family内訳、target selector、正しさ、資源、既知制約を一つの追跡済み要約へ固定する。
- MI300A、MI325X、bare metal、複数GPU、FNUZ FP8のllama.cpp比較、他モデルを完了主張へ含めない。

### 完了結果

- MI300X exact `gfx942:sramecc+:xnack-`、wave64、ROCm 7.14.0でsLLMと固定llama.cppの7行を規定反復で実行し、
  両engineとも7/7 PASSした。sLLMはHIP-only、fallbackなし、反復一致、cleanup 0、資源復帰を満たした。
- fresh profileはGDN `72.72469%`、Full Attention `26.36458%`だった。GDN wave64 column-state v3はoperatorと
  model A/BをPASSし、`10,001/2` prefill中央値を`22.718162442`秒から`6.410255551`秒へ3.54403x改善した。
  全7行candidate再実行は行わず、既定無効のexact gfx942 opt-in候補として`target-separated`に分類した。
- llama.cpp性能同等gateは7行とも未達であり、残差を省略していない。追跡済みMI300X要約は
  [`ci/matrix/phase51-mi300x-summary-v1.json`](../../../../../../ci/matrix/phase51-mi300x-summary-v1.json)、3 target要約は
  [`ci/matrix/three-target-gpu-summary-v1.json`](../../../../../../ci/matrix/three-target-gpu-summary-v1.json)を正本とする。
  gfx1030の長時間2行はsingle final identityではないため行別出典とnon-comparable制約を保持し、gfx1201の100k OOMも維持した。

[Phase 51実行履歴](../../../../../history/2026/08/21-31/phase51-mi300x-application-and-validation.md)

## Phase 52: R9700 100k KV物理コミットOOMの解消

> 状態: complete（2026-08-24）

- exact `gfx1201`、Qwen3.5-4B BF16／FP16 KV、単一要求の`100,000/2`に限定し、自動prefillの実効chunk、
  KV memory kind、K／V／scale各planeのmapped／committed量、VMM page／extent／handle、grow失敗位置を失敗時にも保存する。
- 最優先候補は、複数plane growとcopy-on-writeを一つのtransactionとして扱い、途中失敗時に今回追加・置換した
  mapping／handleだけを完全rollbackする修正である。provider-aware preflight、extent集約、gfx1201長context限定の
  `contiguous-resident`は、取得した原因証拠に応じて個別比較する。
- 明示512はworkspace圧力を分ける限定診断にだけ使い、最終条件にしない。silent retry、CPU/backend fallback、
  要求分割による意味変更は導入しない。
- 最終自動経路で1 warmup＋3 measuredを完走し、生成token、HIP-only、fallback 0、cleanup 0、process消滅、
  HBM／GTT復帰を確認する。llama.cpp性能同等は報告するが完了gateにしない。
- 固定tuple、仮説順、failure injection、証拠schema、停止条件は
  [Phase 52保存済み計画](../../../../archive/2026/08/21-31/phase52-r9700-100k-kv-commit-oom.md)を正本とする。
  Phase 51はこのPhase 52完了時点では一時保留だったが、2026-08-25のユーザー指示で再開して完了した。

### Phase 52完了結果

- exact `gfx1201`のlogical capacity 65,536以上だけを`contiguous-resident`へ固定し、短いcapacityとunknown targetは
  capability-selectedを維持した。runtime OOM後のretry/fallbackは追加していない。
- source `3ed002c476b49417cc702119e37c2389cefb96bc`の自動2K経路で`100,000/2`を1+3回、
  `10,001/2`を3+10回PASSした。全requestで生成`[23066,23066]`、HIP-only、fallback/cleanup 0、資源復帰を確認した。
- 100k E2E中央値は`325.593963905`秒、HBM peakは`15,388,794,880` bytesだった。VMMの内部handle上限値は
  推測せず、総HBM不足ではないprovider physical-commit問題として閉じた。
- 全反復値と物理KV metadataは
  [`phase52-r9700-kv-commit-summary-v1.json`](../../../../../../ci/matrix/phase52-r9700-kv-commit-summary-v1.json)を正とする。

## Phase 39: service operability・認証・observability

> 状態: complete（2026-08-21、host-only。GPU PASS claimなし）

### Scope

- `/healthz`はprocess liveness、`/readyz`はmodel resident・backend受付可否を分離する。
- opt-in Prometheus metricsはqueue、request、token、TTFT、E2E、failure、cancel、VRAM/arena、model identity labelを
  bounded cardinalityで公開し、prompt/token/credentialを含めない。
- read-only props/slots、admin用slot cancel、resumable SSEのevent IDとbounded replay bufferを追加する。
- CORS allowlist、TLS certificate/key、key fileと複数API key、constant-time照合、権限分離したadmin credentialを実装する。

### Acceptance

- startup/loading/ready/draining/failed/shutdownの状態遷移、slow client、disconnect、replay範囲外、queue full、key rotation、
  malformed configをhost integrationで検証する。
- health endpointはGPU処理を起動せず、readinessはfallbackで成功扱いにしない。metric scrapeはgenerationをblockせず、
  label数とmemory上限をtestで固定する。
- TLS/CORS/auth無効時の既存local profileを維持し、有効化時だけ対応surfaceを公開する。

### Closeout

atomic lifecycle、bounded/redacted slot registry、admin cancel、digest-only user/admin key storeとatomic reload、exact CORS、
Rustls TLS、opt-in metrics、nonblocking runtime allocator memory snapshot、明示opt-in resumable SSEを実装した。
既存HTTP contract 10件を含むserver all-target 62件をhostでPASSし、clippy warning 0を確認した。詳細は
[archived Phase plan](../../../../archive/2026/08/21-31/phase39-service-operability.md)と
[history](../../../../../history/2026/08/21-31/phase39-service-operability.md)を正とする。

## Phase 40: token selection・grammar・structured generation

> 状態: complete（2026-08-22）。詳細は
> [archive plan](../../../../archive/2026/08/21-31/phase40-token-selection-grammar-structured-generation.md)を正とする。

### Work units

1. samplerをbackend非依存のordered chainへ型付けし、greedy、temperature、top-p、presence/frequency penalty、seedを
   既存互換のadapterとして移す。
2. top-k、min-p、typical、Mirostat、DRY、XTC、adaptive/dynamic temperature、ignore-EOSを追加し、順序とdefaultを
   versioned request schemaへ固定する。
3. logit bias、選択token logprob、top-logprobsを実装する。NaN/Inf、tie、all-masked、large vocabularyをfail closedにする。
4. GPU samplerはpenalty、mask、partial selection、RNG、selected-token D2Hを一つのprepared pathへまとめる。CPU referenceを
   oracleとして残し、GPUを使えないhost testでfull-model性能を主張しない。
5. GBNFをbounded automatonへcompileし、UTF-8/byte fallbackとtoken trieでvalid-token maskを作る。JSON Schema subsetは
   明示support表へlowerし、unsupported keywordを拒否する。
6. `response_format`、structured output、`n>1` choicesをtransport-independent generationへ接続する。choiceごとのseed、
   KV/sampler/stop stateを分離する。

### Acceptance

- samplerごとにfixed logits、tie、境界値、deterministic seedをreferenceへ一致させ、既存requestのtoken列を維持する。
- grammarは受理文字列だけを生成し、無効schema、状態爆発上限、全token禁止を明示errorにする。
- logprobsは実際にsamplingへ使ったpost-bias/post-mask distributionと一致する。
- GPU pathはexact target、fallbackなし、selected token以外の不要なfull-vocabulary D2Hを行わない。

## Phase 41: prefix/KV・session state・speculation

> 状態: complete（2026-08-22）。詳細なidentity、上限、検証、制約は
> [Phase 41 archive plan](../../../../archive/2026/08/21-31/phase41-prefix-session-speculation.md)を正とする。

### Work units

1. prefix cache keyをmodel-lock fingerprint、adapter identity、renderer/template digest、exact token列、KV encoding、target
   semanticsへ結合し、最長一致とbounded evictionを実装する。
2. vAttention pageをrequest間でread-only共有し、continuation時にcopy-on-writeする。GDN、RoPE position、sampler/stop stateを
   prefix ownerと分離する。
3. context shiftは保持token範囲、absolute/logical position、RoPE scaling、attention maskをversioned policyへ固定する。
4. session/slot checkpointはheader、model/adapter/template identity、token history、KV/GDN/state checksumを持ち、atomic write、
   size/quota、corruption・version mismatch拒否を実装する。KV＋会話＋model SHA-256の簡易永続化をここで満たす。
5. assistant prefillをchat/Responses/Completions共通generation inputへ追加する。
6. external draft modelとngram speculationは同じpropose/verify/accept contractへ接続し、MTPとは別providerとして扱う。
   reject時のCOW rollback、accepted-prefixだけのpublish、通常逐次生成とのtoken一致を維持する。

### Acceptance

- cache hit/miss/partial hit、eviction、concurrent readers、cancel、restart、corrupt/truncated checkpoint、wrong model/adapter/KVを
  検証する。異identityのsilent reuseは不可。
- reused resultはfresh prefillとtoken/visible outputが一致し、cache accounting、cleanup、quotaが閉じる。
- speculation disabled時の既存経路を変えず、有効時はaccepted/rejected/proposed accountingと逐次同一性を示す。

## Phase 42: inference modeと基本public endpoint

状態: 2026-08-22完了。実装・検証の正本は
[archive plan](../../../../archive/2026/08/21-31/phase42-inference-modes-public-endpoints.md)と
[history](../../../../../history/2026/08/21-31/phase42-inference-modes-public-endpoints.md)に移した。

### Scope

- OpenAI Completions、Embeddingsを対応するversioned schema pinへ実装する。
- RerankはOpenAI互換を名乗らず、別のsLLM endpoint/profileとしてquery/document、score意味論、上限を固定する。
- tokenize、detokenize、apply-template、input-token-countを追加し、CLIとHTTPが同じfrontend serviceを使う。
- FIM/infillをmodel capabilityとverified templateへ接続し、unsupported modelは拒否する。

### Acceptance

- tokenizer special token、Unicode、byte fallback、empty/large input、normalization、template digestをCLI/HTTPで一致させる。
- Embeddingsはpooling、normalization、dtype、dimension、usageを明示し、internal embedding gatherをHTTP supportへ正しく接続する。
- Rerank scoreはfixed oracleと順序/tieを満たす。completion/infillはstop、usage、streamingを共有し、`n` choicesは
  Phase 40が所有するchoice stateをwire responseへ写像するだけとする。
- endpoint追加でChat Completions profile v1のreject/response/SSE semanticsを暗黙変更しない。

## Phase 43: Responses・Anthropic Messages・function/tool protocol

> 状態: complete（2026-08-22、host-only。GPU provider変更なし、MI300X実機claimなし）。詳細は
> [archive plan](../../../../archive/2026/08/21-31/phase43-responses-anthropic-tool-protocol.md)と
> [history](../../../../../history/2026/08/21-31/phase43-responses-anthropic-tool-protocol.md)を正とする。

### Work units

1. official Responses schemaを実装開始時のfull commit/versionへpinし、request item、output item、usage、error、stream eventの
   closed state machineを定義する。Chat Completionsのaliasにはしない。
2. Anthropic Messagesはversion header、content block、stop reason、usage、SSE eventを別compatibility profileへ固定する。
3. function/tool definition、tool choice、tool result message、parallel tool call、structured argumentsを共通internal itemへlowerする。
4. Phase 40のJSON Schema grammarをtool argumentsへ適用し、生成後parseだけで正しさを主張しない。
5. reasoning content、assistant prefill、multi-choice、cancel、mid-stream error、resumable eventを各transport adapterへ接続する。
   assistant prefillはPhase 41、multi-choiceはPhase 40、resumable replayはPhase 39のstate machineを再利用する。

### Acceptance

- official-client fixtureとraw HTTPでnon-stream/SSE、tool call/result round trip、structured output、reasoning、cancel、invalid item、
  unsupported multimodal typeを検証する。
- transport間で同じinternal generation requestとvisible token順序を使い、API固有eventへ変換する。
- このPhaseはtool callを生成・受理するが、任意のtool/MCPをserver process内で実行しない。実行はPhase 47まで明示的に無効。

### Closeout

OpenAI Responses `2.3.0` commit `010421dcbd0475277ea8c3e6c1e1cbca4659c4bd`とAnthropic
`2023-06-01`を別profileへ固定し、strict request、non-stream、named SSE、bounded replay、stable ID、usage/stop/errorを
共通schedulerへ接続した。tool definition/choice/call/resultとparallel policyはfrontendのordered internal protocolへ集約し、Qwenだけが
grammar-constrained capabilityを広告する。generated callはclientへ返すdataであり、実行・MCP・外部I/Oは一切追加していない。
machine contract、host grammar/frontend/server unit、raw HTTP/SSE、tool roundtrip、no-execution、redaction、replayとlegacy回帰をPASSした。
Resumable requestは40 output token以下だけをadmitし、成功event batchをPhase 39の64 KiB/event・256 KiB/sessionへ事前適合確認する。

## Phase 44: template・reasoning control・interactive UX

> 状態: complete（2026-08-22、host/frontend/CLI）。MI300X real executionはVM削除済みのためdeferredであり、gfx942
> compile/host evidenceをruntime PASSへ昇格しない。詳細は[Phase 44 archive plan](../../../../archive/2026/08/21-31/phase44-template-reasoning-interactive-ux.md)と
> [Phase 44 history](../../../../../history/2026/08/21-31/phase44-template-reasoning-interactive-ux.md)を正とする。

### Scope

- arbitrary Jinja互換templateとbounded kwargsをsandboxed rendererへ追加する。filesystem、environment、network、process、
  unrestricted object accessは公開しない。
- model lockでreviewed templateをdefaultに保ち、custom templateはdigest付きopt-inとする。
- reasoning budget/mode、生成中のreasoning controlを実装し、Phase 41のassistant-prefill semanticsとPhase 42のFIM/infill
  execution modeをtemplate/CLIへ接続する。Phase 44で別のprefill/FIM stateを作らない。
- interactive CLI、conversation history、reverse prompt、prompt file、save/resume sessionをPhase 41 checkpoint上に実装する。

### Acceptance

- llama.cpp互換fixtureとsLLM canonical templateでrole、special token、tool/reasoning block、Unicode、kwargs、malformed templateを検証する。
- template resource上限、recursion、oversized output、unknown filter/functionをfail closedにする。
- interactive/non-interactive、resume/freshで同じtoken列を生成し、terminal inputとprompt fileを混同しない。

### Closeout

- MiniJinja 2.24.0 exact-pinned generic provider、typed template adapter、digest-bound CLI file reader、bounded kwargs、data-only
  identity reportを実装した。reviewed Qwen default rendererとGemma raw-text pathは暗黙置換せず、raw/Gemma generic inputと未対応backendは
  fail closedとした。
- Reasoning mode/budgetは既存selector・grammar・stop・cancelと同じfrontend controllerへ統合し、1〜4,096 tokenとmulti-token close markerを
  admission前に検証する。Chat/Responses/CLIは同じloweringを使い、Anthropic thinkingはunsupportedのまま残した。
- `chat`はprompt/message/prompt-file/interactive stdinのclosed matrix、regular bounded prompt file、typed transcript、reverse prompt、
  JSONL event、successful-turn-only commitをPhase 41 opaque checkpoint callbackへ接続した。Persistent Qwen chatはhidden reasoning、selected stop、
  matched reverse markerを除外したcanonical history prefixをfresh resident ownerへre-prefillしてopaque captureし、next-turn/fresh-resume exact prefixを維持する。
  load時のexact model/renderer/tokenizer/target/plan/KV identity、conversation+KV pending/current rollback、CLI preflight-before-model-open、
  SIGINT in-flight cancellation laneを実装した。既存`generate`のreport/semanticsは変更していない。
- Phase 44の実装・focused host testsは完了した。MI300X実機correctness/performance、GPU provider/kernel、WebUI、mid-generation/wire session
  resume、Phase 47 tool/MCP executionは本Phaseの完了へ含めず、後続計画またはdeferred laneに残す。

## Phase 45: adapter・control vector・dynamic model lifecycle

Phase 45のhost/API/CLIとRDNA GPU evidenceは完了した。strict `sllm-model-manifest-v1`、offline regular-file/digest preflight、LoRA/control-vector
identity、alias-only lifecycle admin、coalesced registry lease、draining/quarantine/LRU、requestの
`sllm.adapters`/`sllm.control_vectors` extension、CLI `sllm models`を実装し、既存Chat/Completions/Responses/Anthropic semanticsを維持した。
machine profile/schema/validatorとhost contract testsに加え、compact GPU summary/schema/validator/testをCIへ登録済みである。exact
`gfx1030`/`gfx1201` release buildでQwen BF16 disabled/LoRA/control/combinedを各2回bitwise一致でPASSし、HIP-only、fallback=false、resident
`8,411,592,192` bytes、request/workspace baseline復帰、pre/final allocation 0、retryable/quarantine 0を確認した。BroadcastAdd standalone
(`M=1/3`, `H=17`, mismatch 0, cleanup PASS)も両targetでPASSした。gfx942/MI300X runtimeだけをdeferredとし、VM再確保後の別lane入力にする。
詳細は[Phase 45 archive plan](../../../../archive/2026/08/21-31/phase45-adapter-dynamic-model-lifecycle.md)と
[history](../../../../../history/2026/08/21-31/phase45-adapter-dynamic-model-lifecycle.md)を正とする。

### Work units

1. **complete** — preloaded LoRAをverified base model/target tensor/shapeへ結合し、requestごとのadapter setとscaleを指定可能にした。
2. **complete** — control vectorをlayer/range/dtype/scale付きderived artifactとしてlockし、request stateへ適用した。
3. **complete** — model registryを複数alias、lazy load、preload、unload、LRU/cache quota、offline-onlyへ拡張した。
4. **complete** — routerはrequest aliasをimmutable model+adapter identityへ解決し、load中/draining/cancel/failureをPhase 39 readiness/errorへ反映する。
5. **complete** — load/unload中のGPU allocation、in-flight request、shared tokenizer/template、failed model quarantineを所有権contractへ固定した。
6. **complete** — exact RDNA full-model GPU smokeとBroadcastAdd standalone oracleをV620 `gfx1030`/R9700 `gfx1201`でPASS。gfx942/MI300X real executionはdeferred。

### Acceptance

- wrong base、missing tensor、shape/dtype mismatch、duplicate adapter、scale boundary、adapter orderを拒否する。
- adapter/control disabled時はbase logits/tokenを維持し、有効時はbounded slice oracleと両RDNA full-model smokeへ一致する。compact summaryはraw artifactを追跡しない。
- unloadはin-flight ownerを早期解放せず、新規requestを止め、最後のowner後にVRAM/file handleをbaselineへ戻す。

## Phase 46: conversion・quantization・benchmark・品質評価tool

詳細な作業単位、schema、受入条件は
[Phase 46保存済み計画](../../../../archive/2026/08/21-31/phase46-conversion-quantization-benchmark-quality-tools.md)を正本とする。

### Scope

- general HF-to-GGUFをmodel plugin/capability方式へ拡張し、supported architecture/dtypeだけを受理する。
- GGUF split/merge、LoRA conversion、execution-ready layout/repackをmodel-lock/derived-lockへ結合する。
- quantize/imatrixはsLLMが採用するBF16/FP8/NVFP4/MXFP4等だけを対象とし、一般Q8_0/Q4_K対応を導入しない。
- `sllm-bench`、perplexity、KLD、task eval、token/logit/debug dumpを共通dataset/result schemaと再現可能なseedへ固定する。
- FP16 baselineの反復変動から、candidate結果を見る前にKV defaultのdataset、metric、threshold、再測定規則を
  `kv-cache-default-v1` policyへfreezeする。Phase 53は同じpolicy digestをtarget別判定へ使う。

### Acceptance

- converterはtensor catalog、metadata、recipe、source/output digest、tool commit/args/environmentをfail closedに検証する。
- split→mergeはbyte/semantic identity、LoRA conversionはbase+adapter oracle、quantizeはtop1/KLDとbounded slice誤差を記録する。
- benchmarkはwarmup/measurement、E2E/TTFT/TPOT/prefill、model lifecycle、GPU identityを明示し、raw trace/modelを追跡しない。
- debug dumpはopt-in、size上限、secret/prompt方針を持ち、production defaultで無効にする。
- Phase 46は新KV format／HIP kernel／default selectorを実装しない。Phase 53はconverter等のPhase 46全体を待たず、
  default判定に必要なKV policy milestoneだけを依存とする。

## Phase 47: 組込みtool・MCP実行

このPhaseは新しい外部実行security boundaryを作るため、開始時にユーザーがdeployment trust model、許可tool、credential、
network/filesystem、confirmation、audit保持を明示承認するまで`approval-required`とする。Phase番号への割当は実装許可を
先取りしない。

### Proposed scope

- Phase 43のtool callを、server本体から分離したworkerへ渡す。tool allowlist、schema validation、timeout、CPU/memory/output、
  concurrency、cancellationを必須にする。
- MCP client/server connectionはendpointとcapabilityをdeployment設定へpinし、credentialをmodel/promptから分離する。
- network deny-by-default、workspace外filesystem deny、shell/process denyをdefaultとし、capability単位で明示許可する。
- tool resultはuntrusted contentとしてmessageへ戻し、system/developer instructionへ昇格させない。全call/result digest、duration、
  dispositionをsecret-free auditへ残す。

### Acceptance

- schema逸脱、prompt injection、oversized output、timeout、worker crash、disconnect、credential漏洩、path escape、network deny、
  cancel/retryをadversarial testで検証する。
- tool未設定時はPhase 43のtool-call生成だけが利用でき、任意codeを暗黙実行しない。

## Phase 48: minimal WebUI/server UI

2026-08-30に、公開HTTP APIだけを使うprototypeを`webui/`へ実装した。2026-08-31にGPU identityとprefill／decode throughputの
dashboardをdefault viewへ追加し、chatをsecondary viewへ移した。live throughputは単一streaming request前後の`/metrics`差分から
server-level推定値を計算する。`/props.hardware`はserver側HIP device queryだけを公開し、照会不能時は未提供と表示する。demo fixtureを
GPU／live API証拠へ読み替えない。model選択／load／unload、chat SSE、`content`／`reasoning_content`分離、取消し、generation設定、
状態／失敗表示も維持する。
credentialはbrowser memoryだけに保持する。詳細は
[prototype計画](phase48-minimal-webui-prototype.md)と
[実装履歴](../../../../../history/2026/08/21-31/phase48-minimal-webui-prototype.md)を正本とする。
追加承認によりdynamic loopback serverへmodel folder管理を追加し、直下GGUFの互換判定とalias登録を行う。さらに固定`hf` CLI経路で
Hugging FaceのGGUF検索、完全commit SHA付きcommand copy、選択済みfolderへの単一download jobを追加した。これは構造化入力から作る
限定admin操作であり、任意command／destination／token入力やPhase 47の汎用tool実行には拡張しない。
source treeでは`sllm-server`のdefault WebUI起動、無効化／port option、runtime API URL注入、exact CORS、graceful shutdownを統合した。
以下のfull Phase 48 scopeのうちsession、adapter、slot、key status、log、static asset組込み／versioned配布は未着手の継続項目であり、
prototype完了をPhase全体完了へ読み替えない。

### Scope

- sLLM HTTP APIだけを利用する薄いUIとして、GPU identity、prefill／decode throughput、model選択を主surfaceにし、chat/stream、
  reasoning/tool表示、sampling/structured controls、session save/resume、adapter選択、health/metrics要約を提供する。
- admin面はmodel load/unload、slot cancel、key/credentialの値を表示しないstatus、log downloadを権限分離する。
- UI資産はserver binaryへ埋め込むかversioned static artifactとして配布し、CDN依存と外部telemetryをdefaultで持たない。

### Acceptance

- keyboard操作、stream cancel/reconnect、large conversation、tool/reasoning block、upload制限、XSS/CSRF/CORS、auth分離を検証する。
- browserからmodel filesystemへ直接accessせず、承認済みのloopback model-library／固定`hf` admin APIを使う。APIで拒否される操作をUIが迂回しない。
- WebUIのrichnessは完了条件にせず、CLI/APIで利用できる機能の管理surfaceに限定する。

## Phase 53〜54: block16履歴とMXFP8 E4既定化

- Phase 53のdescriptor v1／v2評価とPhase 54のattribution・scale・固定transform研究は完了済みで、詳細は
  [Phase 53保存済み計画](../../../../archive/2026/08/21-31/phase53-kv-fp8-block16-default-adoption.md)と
  [Phase 54保存済み計画](../../../../archive/2026/08/21-31/phase54-kv-fp8-block16-accuracy-research.md)を正本とする。
- block16はMXFP8を上回る品質候補を得られず、2026-08-30のユーザー決定でpublic parser、selector、Qwen graph、Rust→HIP state、
  native state-create ABIから製品経路を廃止した。予約済みABI番号と履歴evidenceだけを監査用に保持する。
- reviewed Qwen3.5-4B BF16 dense textの省略時KVは、exact `gfx1030`、`gfx1201`、`gfx942:sramecc+:xnack-`でstandard OCP
  `kv-mxfp8-e4`（E4M3FN value、block 32、E8M0 scale）とする。対象外model/laneはfixed FP16 recipeを維持し、明示`fp16`をrollbackに残す。
- gfx1030／gfx1201のdirect GPU byte・packed attention oracleはHIP-only、fallback 0、cleanup 0でPASS済みである。
  gfx942のfresh実機証拠は追加のMI300X項目と一括取得し、local RDNA作業をblockしない。

## Phase 62: 再利用可能low-precision block codecとMXFP最適化

- E4M3FN/FNUZ、E5M2、E3M2、E2M1、E8M0のscalar codecと、MX block 32／NV block 16のscale・packed I/Oを
  model／consumer opから独立したdevice-inline primitiveへ抽出する。runtime descriptorはdispatch境界で解決し、
  hot loopはcompile-time specialization、gfx固有実装はtarget trait/providerの内側へ置く。
- Phase 61のMXFP8／MXFP6 matmul、MXFP8 KV append／causal attentionを主要consumerとし、NVFP4も形式差を保持したまま
  意味が一致するcodec/I/Oを再利用する。matmul tile、softmax、KV配置等の演算固有部分は統合しない。
- 共通化段階はbefore/after bit exactとし、W/Aは明示FP16 KV、KVはBF16 weight＋standard OCP MXFP8 E4で性能を分離する。
  persistent FP32 attention/KV planeは追加せず、量子化済みactivationのmaterializeは複数consumerで利益がある範囲だけ採用する。
- exact `gfx1030`／`gfx1201`でmodel-free operatorと固定Qwen3.5-4Bを測り、shared/scoped/rejectedをcandidateごとに決める。
  固定改善率は置かず、Phase 61 MXFP W/Aのdefault、MXFP8 E4 KV default、FP16 rollback、block16廃止を変更しない。
- 固定llama.cpp MXFP4 MMQからQ8_1／int8演算を移さずmulti-column tileだけを参考にしたfollow-upは、両RDNAの固定4B
  prefillをMXFP8で2倍超、MXFP6で約10〜13%改善した。ただしN／formatでoperator結果が逆転するため、col4/col8は
  explicit benchmark-onlyとし、安全なshape selectorを固定するまで既定providerを変更しない。
- 詳細な受入条件、境界値、作業単位、対象外は
  [Phase 62保存済み計画](../../../../archive/2026/08/21-31/phase62-reusable-low-precision-block-optimization.md)を正本とする。

## Phase 63: gfx1201向け大規模MXFP matrix prefill

- exact R9700 `gfx1201`のOCP MXFP8 E4M3 W8A8へmodel非依存WMMA providerを追加し、追加最適化でcontribution LDSを削除、
  N64 tileをM>=128、K>=2,048、1,024<=N<=16,384の整列shapeへscoped defaultにした。M=1、N=32／LM head、gfx1030/gfx942/unknownは
  既存providerを維持する。
- 最終v2 code objectでFP8xFP8-to-FP32 WMMA 8命令、wave32、WG256、LDS 6,912 byte、spill 0を確認した。operator、special E4M3/E8M0、
  同一artifact品質10 case／20 row、512〜4,096 full-modelをHIP-only、fallback false、cleanup 0でPASSした。
- 3 warmup＋10 measuredのprefill中央値は`1,727.595/1,814.619/1,722.844/1,588.366 tok/s`で、v1比約1.75〜1.90倍。
  profileはWMMA v2 71.96%、scope外row8 1.55%、activation quantization 1.78%。persistent BF16/FP32展開やFP32 attention/KV保存は追加していない。
- 数値変更は実数式、FP32 accumulator、BF16 RNEを維持する固定tree変更としてN1へ分類した。top-1 `0.95`は観測として記録し、
  旧KV defaultの`0.99` gateを流用しない。SGLang非copyのclean-room境界も維持した。
- 詳細は
  [Phase 63保存済み計画](../../../../archive/2026/09/1-10/phase63-gfx1201-mxfp-matrix-prefill.md)と
  [追跡要約](../../../../../../ci/matrix/phase63-gfx1201-mxfp8-wmma-prefill-v2.json)を正本とする。

## Phase 66: gfx1201 reusable low-precision providerとattention移植

- architecture、format/block、activation pack、tile、inner productをprepare時にfreezeするmodel非依存providerを
  MXFP8／MXFP6／NVFP4／MXFP4へ接続した。MXFP8 ID37 N128 direct-bothはexact `gfx1201`のPhase 65 familyかつ
  N%128=0へ限定採用し、wide/down operatorをID36比14.45%／6.27%短縮した。
- typed q4k4／q4k8／q8k8 attentionはFP16／MXFP8 E4 KVでbit一致したが、primary M=128/512/2,048が
  4.3〜27.3%遅くproduction不採用とした。BF16 weightでも同じselectorの実dispatchを確認した。
- MXFP6はfull-shapeで必要なtiled16を維持しsmall-N selectorを後続候補、NVFP4／MXFP4 W4A4は既存device kernelへ
  provider routingを移植した。MXFP4 full MoE productionはscope外で、同期A/B用の別W4A4 kernelを追加したとは主張しない。
- 3+10の512〜4,096 MXFP8 prefillは4B `3,249.069405〜3,840.804836 tok/s`、9B
  `1,988.722356〜2,261.647647 tok/s`。persistent BF16/FP32 weight展開やFP32 attention/KVは追加していない。詳細は
  [Phase 66履歴](../../../../../history/2026/09/1-10/phase66-gfx1201-reusable-low-precision-attention-transfer.md)と
  [追跡要約](../../../../../../ci/matrix/phase66-gfx1201-low-precision-provider-summary-v1.json)を正本とする。

## Phase 70: 両RDNA MXFP6のMXFP8実行骨格再利用

- packed E3M2（4 value／3 byte）とblock 32／E8M0 scaleをresident表現のまま維持し、tile ingressでだけ
  E3M2→E4M3 exact変換を行う共通primitiveとtarget別providerを実装した。whole-model E4M3／BF16／FP32展開は追加していない。
- exact `gfx1030` ID43はcol8 MMQを共有しoperator digest一致を確認したが、固定4B prefillが約22%退行したためbenchmark-only。
  exact `gfx1201`はN64 WMMA ID44を初期採用し、追加P70-Fで3-byte groupを一度だけ読むID45へ置換した。ID44比で
  512／2,048-token prefillを1.690／1.608倍へ改善したため同じshapeへ限定採用し、N128 ID46はbenchmark-onlyとした。
- 全64 E3M2 code×4 lane、selector境界、独立FP32 oracle、両target full-model、MXFP8 regression、gfx942 compile-onlyをPASSした。
  詳細は[Phase 70保存済み計画](../../../../archive/2026/09/1-10/phase70-rdna-mxfp6-mxfp8-path-reuse.md)、
  [Phase 70履歴](../../../../../history/2026/09/1-10/phase70-rdna-mxfp6-mxfp8-path-reuse.md)、
  [追跡要約](../../../../../../ci/matrix/phase70-rdna-mxfp6-mxfp8-path-reuse-v1.json)を正本とする。

## Intentional exclusions and deferred items

- Vulkan、一般的なllama.cpp INT4/INT8+scale形式は明示的な製品方針どおり対象外。
- model family/architecture追加、RDNA3等の新hardware、CPU/NVIDIA、parallel/continuous batching、multi-GPU、Infinity Fabric、
  RCCL/RDMAは今回のllama.cpp機能差計画へ含めない。
- LMCache、RadixAttention、Paged Attention、TurboQuant、standard OCP MXFP8 E4以外の残るKV／MX形式は今回のPhaseへ自動追加しない。
  Phase 41のcache/state ABIは後からproviderを追加できる形を維持する。
- README整備、人間による発表、release packagingは別作業であり、Phase 37以降の受入をblockしない。
- fixed llama.cppに存在する機能でも、外部仕様がないrerank、Anthropic、MCP、server extensionは「llama互換」を名乗らず、
  sLLM固有または別仕様pinとして公開する。

## Phase closeout

各Phaseは、採用source、棄却candidate、test/evidence、既知制約、次Phaseへの入力をmatching historyへ記録する。完了または
放棄時にこのplanをarchiveへ移し、main planのroadmap/current stateを更新する。Phase 37以降を一括commitにせず、独立して
review・rollback可能なPhase/work unit単位で公開する。

[全体計画](../../../../main-plan.md) / [対応する履歴](../../../../../history/2026/08/21-31/phase37-plus-mi300x-and-llama-gap-roadmap.md)
