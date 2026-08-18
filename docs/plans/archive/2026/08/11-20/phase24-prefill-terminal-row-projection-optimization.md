# Phase 24: prefill terminal-row projection optimization

> 状態: complete（shared candidate採用）
> 作成日: 2026-08-18

## 初回採否結果と再開決定

- 改訂後の最終判定ではgfx1030/gfx1201のP0/P1/P2/P3/D0全10組でE2E悪化がなく、gfx1030の
  P1/P2/P3はそれぞれ13.14%/12.08%/12.73%改善した。任意pattern 5%以上の条件を満たすためshared candidateを採用した。
- gfx1201も全5 patternで0.09〜0.49%改善し、correctness defectもなかったためtarget別経路は追加していない。
- 通常requestは255 token以上だけをterminal one-row pathへ送り、255未満、明示all-logits、MTP target、MTP draftは
  all-row pathに保つ。これによりshort pathの小さなdescriptor overheadとMTPの行別Argmax/hidden契約を回避した。
- P2のworkspace high-waterは1,149,766,656 bytesから1,023,122,436 bytesへ126,644,220 bytes縮小し、
  model-resident 8,411,592,192 bytesは不変だった。

- 以下は両GPU各5%以上を要求した旧基準での初回結果である。2026-08-18のユーザー指示により旧採用判断を破棄し、
  「全対象patternで安定した悪化がなく、任意のpatternで5%以上改善」を新基準としてPhase 24を再開した。
- shared pathを第一選択とする。exact target別経路はcorrectness defectまたは再現する有意な性能悪化をshared pathで
  解消できない場合だけ導入し、単に片方の改善率が小さいことを分岐理由にしない。

- ユーザーのPhase 24開始指示を受け、5%/2% thresholdとscope境界を実装前に凍結した。
- Qwen normal prefillだけをlast-row projection/Argmaxへ切り替えるbounded candidateを作成し、既存の明示all-logits pathを
  all-rowのまま保持するhost testを通した。
- 256-token / 2-output production laneでは前後baseline平均に対してV620のE2E中央値が12.71%短縮した一方、R9700は
  0.10%短縮に留まり、両GPU5%以上というprimary adoption gateを満たさなかった。
- R9700では`M>1`のhipBLAS GEMMから`M=1`の既存decode reduction kernelへproviderが変わり、不要行削減がwall改善へ
  転化しなかった。profiler observer runではdevice Argmax token未公開も再現した。
- physical logits/workspace allocationも一行へ縮まず、workspace high-waterは1,149,766,656 bytesのままだった。
- 旧基準ではtarget固有provider tuning、fusion、長context救済へ拡張せずcandidateを一度戻した。再開後は同じshared
  last-row candidateを復元し、未実行だったcorrectness/performance matrixとphysical allocation縮小を完了させる。
- 旧採否のbounded結果は`ci/matrix/phase24-terminal-row-summary-v1.json`に残し、再開後の最終結果で同summaryを更新する。

## 目的

Phase 23で最上位となった`P23-O1`を実装・検証する。通常generation prefillは最終tokenのdevice Argmaxと、sampling時の
最終行logitsだけを消費するが、現行Qwen/Gemma graphは全`M`行へvocabulary projectionとArgmaxを実行する。
Phase 24はfinal RMSNorm後の最終行だけをterminal LM headへ渡し、通常prefillの不要な`[M,vocab]`計算とworkspaceを
`[1,vocab]`へ縮小する。

変更はterminal output pathへ限定する。transformer本体、final RMSNorm、attention/GDN、KV/linear state、sampling、MTP hidden、
明示的なall-row logits、model format、public APIは変更しない。Phase 23の`P23-O2` projection-family fusion、`P23-O3`
continuous batching、cold loader、provider tuningを救済策として混ぜない。

## Phase 23から固定する根拠

- Phase 23 bounded summaryは
  `ci/matrix/phase23-performance-discovery-summary-v1.json`、SHA-256
  `dbd928fc8276d8061df1614c26320df3ce610613043e44b9b1ad5f1dee6ece6d`を開始根拠とする。
- Qwen3.5-4B BF16の256-token prefill中央値はV620 2.270 s、R9700 317.58 msで、E1 fixed llama.cpp peerとの
  system gapは6.44x/6.60xだった。
- `[256,248320]` LM-head-shaped workはprofiler device timeのV620 13.48%、R9700 46.92%を占めた。
  production E2Eへ換算したAmdahl上限は13.06%/37.92%、現実的期待改善は8〜13%/20〜38%である。
- frontendのQwen/Gemma `GenerationExecutorV1::prefill`は`token_ids().last()`だけを公開し、sampling pathも
  `last_logits()`一行だけを要求する。
- Qwen graphは通常prefillでもterminal projection/Argmaxを`[M,vocab]` / `[M]`としてlowerする。
  `include_all_logits_bf16`を使うspeculative block pathだけは全行logitsを必要とする。
- Gemma 4 graphもfinal norm後に全行LM head、logit softcap、Argmaxを持つ。ただしPhase 23のGemma profileはM=1だったため、
  M>1での効果量はPhase 24開始時に別途確認する。

## Scope

### Primary対象

- Qwen3.5 dense text generationの通常prefill。
- source modelは固定Qwen3.5-4B BF16、GGUF/model-lockはPhase 23と同じものを使う。
- canonical V620 exact `gfx1030`とR9700 exact `gfx1201`。
- greedy device Argmaxと、temperature/top-p等で最終行logitsを要求するsamplingの両方。
- Qwen multimodal prefillではtext/vision hidden処理を変えず、terminal LM headだけを最終行へ限定する。
- MTP target prefillでは行別Argmaxと全行hidden-state hookを維持するため、terminal LM headも全行のまま保持する。
- terminal rowを表す内部view/selection contract、Qwen execution lowering、必要なhost/GPU oracle、Phase 24 evidence runner。

### Secondary対象

- Gemma 4 dense pathは、R9700の256-token M>1 baselineでterminal projection/softcap/Argmaxがproduction E2Eの5%以上、
  またはdevice timeの10%以上を占めた場合だけ、同じlast-row contractを適用する。
- 上記threshold未満、matched caseを作れない、またはmixed low-bit providerの変更が必要な場合はGemma production変更を行わず、
  Qwen結果と非採用理由だけでPhase 24を完了できる。
- Qwen MoEは共有Qwen graphのhost structural testまでとし、35B full-model GPU rerunをPhase 24の完了条件にしない。

### 非対象

- gate/up、QKV、LM head自体のkernel fusion、shared-load kernel、weight layout変更、new matvec/GEMM provider。
- continuous batching、scheduler、chunked prefill、prefix cache、request state、HTTP/SSE変更。
- GGUF hash/cache、model load、resident allocation、H2D upload pipeline。
- attention/GDN、KV format、TurboQuant、DeepSeek V4、MTP speculation algorithm、vision encoder最適化。
- vocabulary pruning、approximate softmax/argmax、logits quantization、sampling semantics変更。
- all-row logits APIの削除、MTP hidden row削減、public CLI/API flag追加。
- Phase 23全matrixの再実行、vLLM/SGLang環境構築、比較不能なpeer値のratio化。

## 権限とproposal status

- 当初の計画作成指示だけではproduction実装開始やhard gateの承認を含めなかった。その後のユーザーによるPhase 24開始指示で
  初回criteriaを凍結した。2026-08-18の再開指示はその採用部分を明示的に上書きし、全対象patternで安定した悪化なし、
  任意のpatternで5%以上改善、shared path優先を現行criteriaとする。
- proposal originはPhase 23 `P23-O1`、scopeはPhase 24 candidate採否、costはdual-GPU baseline/candidateとbounded Gemma run、
  expiryはP24-A0終了時である。承認後は実装前にcriteriaを凍結し、同一candidate中に緩和しない。
- token/state/all-row保持、範囲外view禁止、fallback禁止は性能process提案ではなくproduction correctness境界であり、
  candidate実装を開始した場合はblockerとして扱う。

## Terminal-output semantic contract

内部契約は実装前に次の二modeへ分離する。名称はprivate実装で変更できるが、意味は変えない。

| request path | terminal projection rows | Argmax rows | logits readback | hidden-state readback |
| --- | ---: | ---: | --- | --- |
| greedy normal prefill `M>=255` | last row 1 | 1 | none | none |
| sampled normal prefill `M>=255` | last row 1 | 1 | last row 1 | none |
| multimodal prefill `M>=255` | last row 1 | 1 | last row 1 | existing contract |
| normal prefill `M<255` | all `M` rows | all `M` rows | optional last row | existing contract |
| MTP target prefill | all `M` rows | all `M` rows | existing contract | all required hidden rows |
| MTP draft graph | all `M` rows | all required rows | existing contract | all required rows |
| normal decode `M=1` | 1、現状同値 | 1 | optional last row | existing contract |
| speculative decode block requiring all logits | all `M` rows | all required rows | all `M` rows | all required rows |
| explicit diagnostic all-logits path | all `M` rows | existing contract | all `M` rows | existing contract |

- last rowはlogical row `M-1`であり、byte offsetはchecked arithmeticで導出する。zero-row、overflow、非contiguous view、
  allocation範囲外はdispatch前にfail closedとする。
- candidateはfinal RMSNormの全行計算を維持し、そのoutputからlast-row alias/viewを作る。Phase 24ではfinal RMSNorm削減を
  追加最適化として混ぜない。
- last-row LM head outputは`[1,vocab]`、Argmax outputは`[1]`とする。frontendがprompt行分のtoken IDsを受け取ることへ
  依存しない。
- Qwenのtied/untied LM head、Gemmaのlogit softcap、quantized model bindingは同じrow policyを伝播し、weight binding、
  tensor scale、softcap数値contractを変えない。
- full-row modeは現行`[M,vocab]` allocation、dispatch、readbackを保持する。last-row modeを理由にall-logitsを暗黙に縮小しない。

## 提案する受入基準（承認後、実装前に凍結）

### Correctness・semantic

1. host graph/layout testは`M=1,2,3,17,255,256,257,2047,2049`を含み、last-row offset、`[1,K]` input、
   `[1,vocab]` logits、`[1]` Argmax、allocation範囲、tied/untied output weightを確認する。
2. distinctive-row tiny GPU oracleは少なくとも`M=2,3,17,255,256,257`を両GPUで実行し、candidateがrow 0や
   `M-2`ではなく`M-1`を射影することを独立f64 referenceで確認する。非有限分類は一致し、fallback、timeout、crashはFAILとする。
3. BF16 last logitsはcandidate実装前にmanifestへ固定する
   `abs(actual-reference) <= 0.015625 + 0.015625 * abs(reference)`を満たす。同じcandidate中のtolerance拡張は禁止する。
   guaranteed-margin caseのtop-1 tokenはexact一致させる。
4. primary full-model greedy caseのprompt/completion token IDs、stop reason、visible output、committed length、KV/linear state length、
   cleanup、fallback auditはbaseline/candidateで一致する。
5. samplingは固定seed、temperature/top-p/penaltyを持つ少なくとも3 caseでtoken列とstop/usageを一致させ、last logitsを上記oracleで
   検証する。差が出る場合は近接logitを理由に無条件許容せず、numeric defectかsampling境界かを分類する。
6. `include_all_logits_bf16`とspeculative blockの`M=2..8`はrow数、順序、全row oracle、accepted/rewind stateを維持する。
   full-row pathがlast-rowへ縮小した場合はcorrectness blockerとする。
7. MTP hidden-state readback、multimodal position/embedding selection、Gemma logit softcapを変更しない。該当pathのhost contractと
   既存focused testをPASSさせる。
8. public API、CLI、GGUF/model lock、supported model/GPU表、default sampling semanticsを変更しない。

### Performance・resource

9. Qwen primary performance laneはnormal profilerを外したproduction設定、同一model/request、各3 warmup + 10 measured以上、
   baseline/candidate交互順、単独GPU可視化で実行する。instrumented/profile wallを採用値にしない。
10. P0/P1/P2/P3/D0の全対象patternと両GPUでstableな性能悪化を残さず、そのうち少なくとも一つのtarget/patternでE2E中央値を
    baseline比5%以上短縮する。改善patternでは対応するprefill/decode spanも短縮し、frontendやcleanup driftだけの差を採用しない。
11. 最初のbracketで0〜2%の悪化が見えるcaseはnoise-neutralと即断せず、candidate/baselineを再度挟んで確認する。最終bracketで
    candidate中央値がbaseline平均を上回る状態が再現するcase、または一回でも2%超悪化するcaseを残さない。decode `M=1`の
    kernel/provider/dispatch identityはbaselineと同じか、差を明示して非悪化を証明する。
12. profile確認では通常256-token prefillのterminal LM head input/output/Argmaxを1 rowとし、`M*vocab`相当のdispatchを残さない。
    full-row controlでは従来どおり`M` rowであることを同じtrace分類で確認する。
13. request/workspace high-waterはbaselineを1%超えて増やさず、last-row logits allocationが`vocab`一行へ縮小したことを
    reportする。model-resident bytesは不変とする。
14. full-model adoptionはcriteria 9〜13を満たし、全pattern非悪化かつ任意pattern 5%以上改善の場合に行う。shared candidateを
    優先し、片方の改善率が5%未満でも非悪化なら共通経路を採用する。exact target別経路はshared pathにcorrectness defectまたは
    再現する性能悪化が残り、分岐がそれを限定的に解消できる場合だけ使用する。

### Evidence・closeout

15. baseline/candidate build input、semantic tree、binary SHA-256、ROCm、target、model/derived lock、GPU identity、health、
    exact prompt/output条件をreportへ固定する。draftはdirty treeを許容するが、integration evidenceはsemantic identityを明示する。
16. raw model、binary、full logits、生成全文、rocprof traceを追跡しない。Git管理するのはschema、bounded aggregate、digest、
    runner、plan/historyだけとする。
17. affected host/tool checks、dual-GPU oracle/performance、1回のintegration review、findingのfocused re-review、main plan/history/
    provenance consistencyを完了する。candidateを棄却した場合も否定結果を残してPhase 24を完了できる。

## 計測case

| case | input / output | 用途 | hard performance判定 |
| --- | --- | --- | --- |
| H0 | synthetic `M=1,2,3,17,255,256,257,2047,2049` | checked row view/layout | 不可 |
| G0 | distinctive-row projection `M=2,3,17,255,256,257` | last-row numeric oracle | 不可 |
| P0 | Qwen normal prefill 17 / 2 | short non-aligned regression | 悪化なし |
| P1 | Qwen normal prefill 255 / 2 | one-row境界 | 悪化なし、任意case 5%判定対象 |
| P2 | Qwen normal prefill 256 / 2 | primary prefill | 悪化なし、任意case 5%判定対象 |
| P3 | Qwen normal prefill 257 / 2 | boundary直後 | 悪化なし、任意case 5%判定対象 |
| D0 | Qwen 28 / 128 | M=1 decode regression | 悪化なし |
| S0 | Qwen sampled、3 fixed profiles | last-logits/sampling保持 | 性能claim不可 |
| A0 | Qwen all-logits/MTP block `M=2..8` | full-row control | 性能claim不可 |
| X0 | Qwen multimodal host/focused control | hidden/position保持 | 性能claim不可 |
| G1 | Gemma 256 / 2、R9700 | secondary inclusion gate | scope判定のみ |

- P0/P1はone-row閾値の両側、P2/P3は連続するnon-aligned caseを確認する。全pattern非悪化と任意pattern 5%以上を
  P0/P1/P2/P3/D0の固定matrixだけで判定し、2K caseを追加してcandidateを救済しない。
- H0/G0はpower-of-twoだけにせず、`M=3,17,255,257,2049`を必須とする。
- baseline/candidateは同じGPUへ同時常駐させず、run順を交互化する。GPU間の絶対値ではなく各GPU内のpaired差を採否へ使う。

## 実装方針

- 最小変更はQwen execution lowering/layoutでfinal RMSNorm outputのlast-row aliasをterminal projection inputへ結び、
  logits/Argmax output viewを一行へすることとする。新しいHIP kernelやpublic semantic op kindは追加しない。
- row policyはlabel文字列だけに依存させず、terminal output node/binding classまたはprivate typed contractへ結び付ける。
- checked offset/extent helperをQwen/Gemmaで安全に共有できる場合だけ共通化する。共通化のためにmodel-neutral runtime ABIを
  拡張しない。
- prepared execution identity/cache keyはrow policyを含め、all-row planをlast-row planとして再利用しない。
- baseline source/binaryをcandidate実装前に固定する。candidate defaultは採用criteriaを通過するまでcommit対象とせず、
  不採用時はcandidate sourceを除去してbaseline behaviorへ戻す。

## 作業順序

### P24-A0: contract、baseline、tolerance固定

- Phase 23 summary/digest、current source/tree、Qwen/Gemma/MTP/multimodal path、既存testを棚卸しする。
- baseline binary/build identityを両targetで固定し、P0/P1/P2/P3/D0のfresh baselineを取得する。
- last-row/all-row mode、offset arithmetic、numeric tolerance、case token IDs、sampling profiles、adoption thresholdをschema/manifestへ固定する。
- Gemma G1をR9700で一度取得し、secondary scope gateを閉じる。結果を見る前に5% E2Eまたは10% device share thresholdを変更しない。

### P24-A1: host structural contract

- final RMSNorm後のchecked last-row view helper/private contractを追加し、H0の全shapeでoffset、extent、contiguity、overflowを検証する。
- Qwen tied/untied graph、all-row path、prepared identityをhost testで分離する。
- Gemmaをscopeへ含める場合はsoftcapを含むterminal chainへrow policyが一貫して伝播するhost testを先に追加する。

### P24-A2: Qwen bounded implementation

- normal prefillだけをlast-row projection/Argmaxへlowerする。decode M=1とall-row controlは意味上同じ経路を維持する。
- last logits、MTP hidden、multimodal state、transactional publication、cleanupを既存順序のまま保持する。
- G0、S0、A0、X0を通すまでfull-model performance claimを行わない。

### P24-A3: dual-GPU correctnessとmechanism proof

- exact target binaryでG0を両GPU実行し、HIP-only、fallbackなし、numerical oracle、last-row dispatch、cleanupを確認する。
- P2 normal pathとA0 all-row controlをprofileし、1-row化とfull-row維持を別runで証明する。
- correctness failure時はperformance測定へ進まず、candidateを修正または棄却する。

### P24-A4: counterbalanced full-model採否

- P0/P1/P2/P3/D0をbaseline/candidate交互順で実行し、median、MAD、paired差、health、VRAM high-waterを集計する。
- criteria 10〜14でshared candidateを採否する。局所operator改善だけをPhase 24成功とせず、固定10組のproduction E2Eで判断する。
- 不採用時はcandidate source/defaultを残さず、test/evidence/否定結果だけを保持する。

### P24-A5: bounded Gemma extension

- A0 gateを通った場合だけ同じlast-row contractをGemmaへ適用し、host contract、R9700 G0/P2相当、sampling/softcap、cleanupを確認する。
- Qwen採用をGemma最適化の成功へ依存させない。Gemma固有provider tuningが必要なら`P23-O2`系follow-upへ戻す。

### P24-A6: integration、review、closeout

- affected test、format、schema/manifest、Markdown/provenance、dual-GPU final evidenceを実行する。
- integration reviewではoff-by-one view、all-row縮小、prepared-plan混同、MTP/multimodal state、fallback、cleanup、
  baseline/candidate identity、performance arithmeticを確認する。
- main plan/historyを実結果へ同期し、本planをarchiveする。`P23-O2`または`P23-O3`を自動開始しない。

## Finding分類

- correctness/security blocker: wrong row、out-of-range alias、all-row loss、token/state/cleanup差、fallback、poison/publication破壊。
- release evidence: immutable final identity、dual-GPU oracle、counterbalanced P2/D0、health、bounded aggregate。
- process improvement: runner/schema再利用、追加case、より良いprofile分類。採用条件を後から増やさない。
- optional hardening: additional model/GPU/context、long soak、extra sampling profile。
- style/docs: naming、説明、非blockingなrefactor。

## 最終結果

- `TerminalOutputRows::{Last, All}`をprivate typed contractとして導入し、gfx1030/gfx1201で同一loweringを使用した。
  通常requestの`M>=255`だけをlast-row projection/Argmaxへ切り替え、all-logits、MTP target/draft、`M<255`はall-rowを維持した。
- host focused testは19/19 PASS。distinctive-row GPU oracleは両targetで`M=2,3,17,255,256,257`のlogical `M-1`を
  f64 referenceと最大絶対誤差0で選択した。sampling 3 profile、MTP幅2、token/stop/usage/audit/cleanupも両targetで一致した。
- profilerはP2のterminal projectionを一行の`[1,248320]`、Argmaxを一行として確認し、physical logits/Argmax allocationも
  one-rowへ縮小した。旧observer failureはMTP targetをall-rowへ戻すことで解消した。
- E2E改善率はgfx1030がP0/P1/P2/P3/D0=`0.14/13.14/12.08/12.73/0.17%`、gfx1201が
  `0.32/0.40/0.49/0.35/0.09%`。全組非悪化かつ任意組5%以上を満たし、target分岐なしで採用した。
- Gemma拡張はQwen primaryの採用に不要で、bounded scopeへ別最適化を混ぜないため実行しなかった。
- immutable digest、aggregate、correctness、profile、memory、最終dispositionはbounded summaryを正とする。

## Rollback・停止・再計画

- correctness criterionが一つでも失敗した場合はcandidateをdefaultへ採用しない。performanceでcorrectnessを相殺しない。
- 全patternのいずれにも5%以上改善がない、またはstableな悪化が残る場合は、Phase 24中にkernel fusion、final RMSNorm削減、
  2K-only救済を追加しない。shared pathの問題がexact targetへ限定される場合だけ最小のtarget分岐を検討する。
- M=1 decodeが悪化する場合はprepared provider/cache identityを調べ、再計測と2回の修正でも解消しなければcandidateを棄却する。
- all-row/MTP/multimodal契約をlast-row modeと安全に分離できない場合はgraph-wide変更へ拡張せず、Qwen通常prefillだけの
  private binding案へ縮小する。それも成立しなければPhase 24を否定結果で閉じる。
- Gemma gate不成立はPhase 24 blockerではない。Qwen primaryを継続し、Gemmaは理由付きfollow-upへ戻す。
- 同じwork unitが2回reject、review時間が実装時間超、functional progressが1時間停止、verification/docsが30%超、
  見積り1.5倍超、acceptance/gate変更時は作業を止めて同じunitを再計画する。

[Phase 23 bounded summary](../../../../../../ci/matrix/phase23-performance-discovery-summary-v1.json)
[Phase 23 technical note](../../../../../references/phase23-inference-engine-performance-differential.md)
[bounded summary](../../../../../../ci/matrix/phase24-terminal-row-summary-v1.json)
[対応する履歴](../../../../../history/2026/08/11-20/phase24-prefill-terminal-row-projection-optimization.md)
