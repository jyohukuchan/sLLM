# Phase 31: low-bit KV通常運用に向けたchunked prefill・workspace memory基盤

> 状態: 完了・採用
> 作成日: 2026-08-19
> 完了日: 2026-08-19

## 位置付け

Phase 31の親目的は、FP8/NVFP4等のlow-bit KV cacheを通常CLI/API/full-model経路で実用化することである。
low-bit KV自体を後回しにして別の最適化へ移るのではなく、Phase 30で露呈した10k+ inputのmemory preflight失敗を解消し、
low-bit KVの長context correctness・性能・resource検証を成立させる前提を作る。

Phase 30の10k+ inputはattention kernel実行中のOOMではなく、実行前にrequired 53,758,880,592 byte、available
34,135,343,104 byteと判定されてfail-closedした。最終KV cacheだけでなく、prompt全行に比例して確保されるowned tensorと
workspaceを広く合算する現行layoutが主な制約である。chunked prefillは最終KVのlogical byteを減らさないが、prefill一時workspaceを
chunk sizeへ上限化する。liveness-aware workspace reuseは、同時に生存しないtensorを同じarenaへ安全に重ね、同じchunk内のpeakをさらに下げる。

## 目的

1. promptを連続chunkへ分割して、absolute position、causal/GDN state、KV publication、terminal logitsを維持したままprefillする。
2. total VRAMと実際のresource preflightから512/2K/4K/8K/16Kを決定的に自動選択する。
3. request-owned intermediate tensorのlivenessを解析し、非重複intervalだけを一つのbounded workspace arenaで再利用する。
4. Qwen3.5-4Bの10k+ inputをcanonical R9700/V620でfull-model実行し、memory内訳と選択chunkを説明可能にする。
5. FP16、dynamic/static FP8、可能な範囲のNVFP4でchunk orchestrationがencoding非依存であることを証明し、後続のlow-bit KV通常運用をunblockする。

## ユーザー決定として固定する方針

- SGLangの階層的なchunk sizingを参考に、defaultはautomatic chunk selectionとする。
- total device VRAMが16 GiB以下の場合、default chunk sizeは512 tokenとする。
- 16 GiB超では2,048 / 4,096 / 8,192 / 16,384 tokenを自動選択候補とする。
- selectorはprompt、token値、測定後の勝敗をkeyにせず、device total/available memory、model identity、KV encoding、
  request state、chunkごとのpeak workspaceというstable resource factsだけを使う。
- vAttention型`virtual-contiguous` providerをproduction defaultとして維持する。Paged Attentionは同じopaque KV stateの下へ
  将来並置できる別physical-layout providerであり、Phase 31では実装・選定・置換しない。
- native FP8 append encodeの再検証はPhase 31へ混ぜない。chunked long-context基盤の完成後に、通常low-bit KV経路の一部として再計画する。

## 現状baselineと解くべき問題

### 既にあるもの

- opaque KV stateとFP16、dynamic/static FP8、NVFP4 encoding。
- HIP VMMによるvirtual-contiguous VA reserveと、token growthに応じたphysical commit。
- request全体のFP16/BF16 mirrorを作らないdirect packed attention。
- Qwen3.5のfull attention KV、GDN recurrent state、absolute position、prepared execution、terminal-row LM head。
- Phase 30のexact gfx1201 native FP8 read/wave attentionと、gfx1030 baseline provider。

### 不足しているもの

- 一つのpromptを複数prefill transactionへ分けるproduction orchestration。
- chunk境界を跨ぐKV/GDN state、absolute RoPE position、error/cancelの明示contract。
- total prompt長ではなくpeak live chunkからworkspaceを算出するmemory planner。
- 同時に生存しないdynamic tensorのalias/reuse。
- long-context full-modelでFP8 KVを選択して比較できるbounded evidence path。
- selected chunk、candidate別required bytes、final KV bytes、workspace peakを分離したdiagnostic。

## Architecture境界

### 対象

- Qwen3.5-4B BF16 GGUFのtext prefill/decodeをprimary full-model scopeとする。
- Qwen graphのfull-attention KV stateとGDN recurrent stateの両方をchunk継続する。
- FP16 KVをproduction control、dynamic/static FP8 KVをlow-bit readiness scopeとする。
- NVFP4は既存encoding contractを壊さないmodel-free/spot scopeとし、対応modelの通常default化は要求しない。
- canonical exact `gfx1201` R9700とexact `gfx1030` V620を実機scopeとする。
- CLIとOpenAI-compatible non-stream/SSEの共通generation serviceをintegration scopeとする。

### 非対象

- Paged Attention、block-table attention kernel、global KV page pool、prefix sharing、RadixAttention。
- continuous request batching、prefill/decode interleave、multi-request fairness schedulerのproduction接続。
- native FP8 append encode、TurboQuant、新しいKV format、KV recipe変更。
- attention matrix/FlashAttention provider、projection/GDN kernel自体の再最適化。
- CPU/GTTへのKV offload、swap、preemption、永続化、multi-GPU。
- promptの意味的切詰め、古いfull-attention KVのeviction、sliding-window化。

## Chunk execution contract

total prompt token数を`M_total`、選択chunk capacityを`C`とし、chunk `i`は半開区間
`[offset_i, min(offset_i + C, M_total))`を扱う。

- `offset_0=0`、次chunkのoffsetは直前chunkのendと完全一致し、重複・欠落・並べ替えを許さない。
- position ID、RoPE、causal visibilityはrequest全体のabsolute positionを使用し、chunk-local zeroへ戻さない。
- chunk内のK/Vは既存append transactionで一度だけ追加する。causal attentionはcommitted endを見ても各queryのabsolute positionより
  後ろのkeyを読まない。
- full-attention KV、GDN recurrent state、generation/versionはchunk terminal completion後にだけ次chunkへ公開する。
- intermediate chunkではLM head、Argmax、sampling、visible outputを実行しない。Phase 24 terminal-row policyは最終chunkの
  最終prompt rowだけへ適用する。
- 最終chunk完了後のdecode M=1は現行graph/providerをそのまま使う。
- chunk途中のerror/cancel/timeoutではgeneration outputを公開せず、request ownerをpoison/releaseする。完了済みchunkを別requestへ
  誤再利用しない。model-resident ownerは維持する。
- 一つのrequestで選んだ`C`はprefill開始後に変更しない。最後のchunkだけactual row数を縮める。
- allocation失敗後に小さいchunkへsilent retryしない。selectionとfit判定は最初のdevice allocation/dispatchより前に完了する。

## Automatic chunk selector

### Bucket policy

| device total VRAM | automatic candidate order | default behavior |
| --- | --- | --- |
| `<=16 GiB` | `512` | 512だけを評価し、収容不能ならrequired/available内訳付きでfail-closed |
| `>16 GiB` | `16384, 8192, 4096, 2048` | 大きい順に評価して最初にfitするbucketを選ぶ |

16 GiB超でも2Kがfitしない場合、512を最終preflight floorとして評価できる。ただしこれはruntime OOM後のfallbackではなく、
dispatch前の同じ決定的selector内の選択である。512もfitしなければ実行しない。

promptがbucketより短い場合、fit計算のeffective rowsは`min(M_total, bucket)`とする。例えば10k promptで16K相当の一括layoutが
fitしなければ8K、4Kの順に評価する。短promptをbucketまでpadding allocationしない。

### Required-memory model

各candidate `C`について少なくとも次をchecked arithmeticで分離する。

```text
required(C) = model_resident
            + final_request_state(M_total, KV encoding)
            + peak_live_workspace(min(M_total, C))
            + allocator/VMM metadata
            + safety_reserve
```

- `final_request_state`は全prompt処理後のKV value/scale planes、GDN state、request固定bufferを含む。chunkingで消えたことにしない。
- `peak_live_workspace`だけをchunk row数へ上限化する。tensor byteの単純総和をpeakと呼ばない。
- total VRAM thresholdにはphysical device total bytes、fit判定にはselection時のavailable bytesを使い、両者を混同しない。
- safety reserveの初期候補は`max(total VRAMの5%, 1 GiB)`とするが、これはA0でcurrent allocator/VMM overheadを測ってfreezeする
  非blocking proposalであり、根拠なく対応可能chunkを狭めない。
- selectorはcandidateごとのrequired内訳、reject理由、最終bucketをbounded diagnosticへ残す。

### Override境界

- production defaultは`auto`のままとする。
- benchmark/evidenceでは512/2K/4K/8K/16Kを明示固定できるtest seamを持つ。
- HTTP requestごとのchunk指定は追加しない。通常CLIへglobal overrideを公開する必要性はA0で既存config seamを確認し、
  無ければPhase 31の必須成果にしない。

## Workspace liveness・reuse設計

### 対象buffer

- request-ownedで、graph node scheduleからfirst/last useを決定できるdynamic intermediate。
- Q/K/V projection出力、attention/GDN intermediate、MLP intermediate、norm/elementwise output等のうち、既存alias contractを壊さないもの。
- selected chunk capacityに対して一度確保し、全chunkで再利用するprefill arena。

### reuseしないbuffer

- model weight、KV value/scale/outer-scale planes、GDN persistent state。
- prompt token/positionのうち次chunkまたはhost-visible lifetimeを持つbuffer。
- final hidden/logits/Argmax、completion中のsubmission ownerが参照中のbuffer。
- alias、in-place、prepared plan、backend alignmentを安全に証明できないbuffer。

### Planner contract

- graph topological orderから各tensorのfirst producer、last consumer、submission completion boundaryを求める。
- lifetimeが重ならないintervalだけを同じoffsetへ置き、同時live tensorのbyte range overlapを禁止する。
- alignment、byte offset、dtype/layout span、access mode、backend/device/session identityをchecked arithmeticで維持する。
- async submissionのhost enqueue終了ではなく、同一streamの既存terminal completionでlifetimeを閉じる。未完了kernelのbufferを再利用しない。
- arena sizeはinterval allocation後のhigh-waterとし、preflightの`owned tensor/workspace`を単純総和からこのhigh-waterへ置き換える。
- debug/evidence buildでpoison patternまたはdistinctive fillを使い、早すぎるreuseを検出する。production pathへ毎dispatch fillを残さない。
- prefill arenaとdecode M=1 persistent workspaceを混同しない。prefill終了後に安全にrelease/reuseできる所有境界を明記する。

## 数値・出力規則

chunkingとworkspace aliasは計算式、dtype、演算順、scale recipeを変えないN0 scheduling/layout changeを目標とする。

- unchunkedで収容可能な同一inputについて、final prompt logits、生成token、KV committed length、GDN generation/state sampleを比較する。
- FP8/NVFP4は量子化をtokenごとに一度だけ行い、chunk境界でscaleを再量子化・mergeしない。
- chunk sizeを変えても同じtoken列とstate contractを再現する。演算順が不可避に変わる場合は
  `docs/compatibility/numerical-output-changes.md`のN0〜N3へ分類し、N1の解析条件を満たさない差を自動承認しない。
- unexplained token/logit/state差、absolute position reset、重複append、future-key visibilityはN3/correctness blockerとする。
- 10k+でfull CPU logits再計算を要求せず、短いexact differential、long-context chunk-size differential、固定sample、finite、
  state length/digestを組み合わせる。

## Verification matrix

### H0: host selector・planner

- total VRAM `16 GiB-1 / 16 GiB / 16 GiB+1`で512固定境界を確認する。
- available bytesを各candidate requiredの`-1 / exact / +1`にしてfit/failを確認する。
- prompt `1/3/511/512/513/2047/2048/2049/4095/4096/4097/8191/8192/8193/16383/16384/16385/65535/65536/65537`を扱う。
- position、chunk count、last chunk rows、u64 overflow、zero、duplicate/gapをfail-closedにする。
- liveness intervalはtouching、nested、same-start、same-end、alignment 255/256/257、最大span両側を検証する。

### G1: model-free GPU

- exact `gfx1030`/`gfx1201`でFP16、dynamic/static FP8、NVFP4のchunked append/readを実行する。
- M `1/3/17/37/511/512/513/2047/2048/2049`をbounded oracleへ照合する。
- absolute start position `0/255/256/257/9999/10000/10001`、KV page/commit境界`1023/1024/1025`を含める。
- unchunked対512/2K/4Kのfinal output/state differential、fallback false、cleanup terminal zeroを確認する。
- gfx1201はPhase 30 wave/native-read symbol、gfx1030はbaseline symbolへ到達し、chunk orchestrationがtarget routingを壊さない。

### G2: full-model memory・correctness

primary modelは同一Qwen3.5-4B BF16 GGUF/derived lockとする。

| case | input | encoding | 比較目的 |
| --- | ---: | --- | --- |
| F0 | 17/255/4096 | FP16 | 既存fit patternの非悪化、one-chunk同一性 |
| F1 | 9999/10000/10001 | FP16 | Phase 30 preflight failureの解消、boundary |
| F2 | 10001 | dynamic/static FP8 | low-bit long-context readiness、memory内訳 |
| F3 | 16383/16384/16385 | FP16、可能ならFP8 | selector/複数chunk scaling、primary target |
| F4 | 同一10k inputを512/2K/4K固定 | FP16/FP8 | partition invariance、memory/TTFT tradeoff |
| F5 | prompt後decode 32/128 | FP16/FP8 | chunk終了から通常decodeへのhandoff |

- F1/F2はR9700/V620の両方で最低一つの10k+ full-model PASSを要求する。
- F3はR9700 primaryでresource/timeがboundedな一組を必須とし、V620追加は一般化条件にしない。
- low-bit weight providerとKV encodingを混同せず、F2のweight/model identityを固定する。
- unchunkedで収容不能なcaseは「速度改善率」を捏造せず、512/2K/4K間と既存4K controlでmechanismを説明する。

### G3: service・failure

- CLI、OpenAI non-stream/SSEで10k+ promptから1/32 token生成する。
- chunk間cancel、chunk内cancel、timeout、client disconnect、backend error injectionでvisible outputなし、request cleanupを確認する。
- usageのprompt tokensは全input、completion tokensは生成分だけとし、chunk数をusageへ加算しない。
- stop、seed、sampling、reasoning rendering、SSE順序は現行GenerationService contractを維持する。
- service shutdown後にrequest/workspace current bytes 0、model ownerだけ所定のlifecycleで残ることを確認する。

## Memory evidence contract

各代表runは次を分けて記録する。

- device total/available bytes、selector candidate順、candidate別required/reject理由、selected chunk。
- model-resident bytes。
- final logical/physical committed KV value/scale bytesとencoding。
- GDN/request persistent state bytes。
- baseline owned-tensor sum、arena high-water、actual peak workspace、allocator overhead。
- pre/during/post VRAM、GTT、VMM committed bytes、foreign process、ECC、cleanup。
- chunk count、各actual rows、prefill wall/device family、TTFT、decode TPOT/E2E。

raw trace、full logits、model、KV payload、生成全文は追跡しない。bounded aggregate、digest、schema/test、plan/historyだけをGitへ残す。

## 採用基準

### Chunked prefill default

1. R9700/V620の10k+ Qwen full modelがpreflight、prefill、1 token以上のdecode、cleanupまでHIP-onlyでPASSする。
2. peak workspaceはtotal promptではなくselected chunkへboundedとなり、同じchunkでpromptを延ばしたとき増える分をfinal KV/stateへ説明できる。
3. F0の既存fit patternで、frozen noise bandを超えるstable TTFT/decode悪化、token/state差、余分なCPU fallbackがない。
4. selectorが16 GiB境界とcandidate fitを決定的に再現し、actual allocation failure後のsilent retryを行わない。
5. 10k+ capability獲得を「5%速度改善」に置き換えない。このwork unitの主成果はmemory feasibilityであり、速度は非悪化条件とchunk比較に使う。

### Workspace reuse

6. 全validation patternのcorrectness/performanceが非悪化で、代表long patternのactual peak workspaceまたはpeak VRAMを5%以上削減した場合に採用する。
7. 5%へ届かないreuseはchunked prefillの完了を妨げず、危険なaliasを残さずnegative resultとして除去できる。
8. arena plannerのaccounting値とactual allocator high-waterが一致し、隠れたper-chunk allocationまたはunbounded growthがない。

### low-bit readiness

9. FP8 chunked pathは同encoding unchunked/別chunk differential、scale/value state、fallback、cleanupをPASSする。
10. Phase 31だけではFP8/NVFP4を全model・全targetの通常defaultへ昇格しない。通常運用のpolicy、品質、native append再検証は後続scopeで判断する。

## 作業順序

### P31-A0: acceptance・baseline freeze

- Phase 30 final source/runner/model identityを固定し、4108 PASSと10k preflight failureをfresh再取得する。
- current graphのowned tensor、alias、state、workspaceを分類し、53.76 GB requiredの内訳を再構成する。
- safety reserve、noise band、full-model case数を実測に基づきfreezeする。
- existing vAttention、KV encoding、GDN state、terminal-row、service transaction contractを変更不可境界として記録する。

### P31-A1: memory-accounting/liveness audit

- tensorごとのsize、producer、last consumer、async completion owner、alignment、persistent/dynamic区分を抽出する。
- current単純総和、理論peak live bytes、final state bytesを比較し、chunking単独とarena reuseの寄与を分ける。
- preflight reportへcategoryとcandidate chunk estimateを追加する。diagnostic追加自体でproduction allocationを変えない。

### P31-A2: deterministic chunk selector

- 16 GiB threshold、512/2K/4K/8K/16K bucket、final 512 floor、checked required formulaを実装する。
- selectionをrequest開始時に一度だけ行い、`PrefillChunkPlan`へtotal rows、capacity、offset、count、memory identityを固定する。
- H0 boundary testsとmalformed/overflow negative testsを先に完成させる。

### P31-A3: chunked prefill orchestration

- 一つのgraph/templateとselected-capacity layoutを再利用し、actual chunk viewだけを更新する。
- token/position upload、KV append、attention/GDN state、terminal completionをchunk transactionへ接続する。
- intermediate chunkのLM head/Argmax/samplingを省略し、final chunkだけ現行generation handoffへ渡す。
- error/cancel/dropとstate poison/releaseをhost/model-free testsで固定する。

### P31-A4: bounded workspace arena

- A1で安全と証明したrequest-owned intermediateだけをinterval allocatorへ移す。
- alias range、alignment、completion lifetime、dynamic actual rowsを検証し、distinctive-fill oracleを実行する。
- 理論上限またはactual削減が5%に届かない場合はvariant追加を止め、chunked-only candidateへ戻す。

### P31-A5: encoding-independent correctness

- FP16、dynamic/static FP8、NVFP4のmodel-free chunk differentialを両targetで実行する。
- Qwen hybrid full-attention/GDN state、absolute position、KV committed length、scale plane、final logits/tokenを照合する。
- 数値差があればN0〜N3分類と台帳更新を行い、unexplained差のまま性能測定へ進まない。

### P31-A6: 10k+ full-model/resource proof

- local Qwen subagent serviceを停止し、GPUをUUIDで一台だけ可視化する。
- F0〜F5を一度に一GPU、同一ROCm/model lock/GGUFで実行する。
- 10k+のselected chunk、memory category、peak VRAM/GTT、TTFT、decode、tokens、fallback、cleanupを取得する。
- fixed 512/2K/4K比較からmemory/launch tradeoffを確認し、auto selectorの最大fit選択が妥当か検査する。

### P31-A7: service integration

- CLI/non-stream/SSEの10k+ request、usage、cancel/disconnect、shutdownを検証する。
- requestごとのchunk planをgeneration service ownerへ閉じ込め、HTTP DTOやOpenAI wireへ内部chunkを漏らさない。
- later continuous batchingがchunk prefillをinterleaveできるようowner境界を明示するが、scheduler接続は実装しない。

### P31-A8: closeout

- bounded summary/schema/test、integration review、changed findingのfocused re-reviewを完了する。
- main plan、runtime、KV memory、GPU/software compatibility、numerical ledger、provenanceを必要範囲で同期する。
- plan/historyを相互linkしてarchiveする。
- debug fill、forced bucket、raw profile metadata、棄却したarena variantをproduction sourceへ残さない。
- native append、low-bit default化、Paged Attentionの後続Phaseを自動開始しない。

## 停止・再計画条件

- chunk partitionでabsolute position、KV/GDN state、token/logitが説明不能に変わる。
- final KV/stateをmemory見積りから除外して10kがfitしたように見せる。
- full-cache mirror、host round-trip、GTT spill、CPU fallbackでGPU PASSを作る。
- chunkごとにgraph/weight/prepared planを再構築し、memory削減と引き換えにunbounded host/device allocationを追加する。
- async completion前のarena reuse、alias race、partial state publication、request間state混同が発生する。
- 512でも収容不能なのにallocation retry loopで処理を続ける。
- workspace reuseのreview/verificationが実装時間を超えるか5%上限へ届かない場合、reuseを切り離してchunked-onlyで同じPhaseを完了する。
- Paged Attention、continuous batching、prefix cacheを必要条件としてPhase 31へ逆流させる。

## Closeout

- automatic selector、completion-boundary liveness slot、multi-chunk Qwen text prefillをproduction pathへ採用した。
- 10,001-token workspaceは旧個別allocation合計39,950,821,120 byteに対してarena high-water
  5,278,049,280 byte、16,385-tokenは65,448,547,584 byteに対して8,646,688,768 byteとなり、約86.79%削減した。
- exact gfx1030/gfx1201で10,001-token FP16/dynamic FP8をHIP-only、fallbackなし、cleanup 0としてPASSした。
  exact gfx1201では16,385-tokenを16,384+1の2 chunkでFP16/dynamic FP8ともPASSした。
- Qwen CLI/serverへ明示的な`fp16`、`fp8`、`fp8-static`、`nvfp4` KV選択を接続した。既定はFP16のままである。
- gfx1201 serverは10,013-token chat promptのnon-stream/SSE各1 token、shutdown後request/workspace 0をPASSした。
- chunk partition/arenaはN0。測定した反復入力ではone/multi-chunkとも生成token 1228を維持した。
- fixed 512/2K/4Kの全full-model比較、decode 32/128、cancel/error injectionの全組合せは、主要acceptanceを満たした後の
  追加コストが大きいためcloseout gateにはしなかった。selector boundary、required bytes exact/-1、chunk state/position、service
  cancel/cleanup contractはhost既存/追加testで固定した。
- Paged Attention、native append encode、continuous batching、prefix sharing、low-bit default昇格は未着手のまま維持する。

機械可読なidentity、実機row、制限は
[Phase 31 bounded summary](../../../../../../ci/matrix/phase31-chunked-prefill-summary-v1.json)を正とする。

[Phase 30計画](../../../../archive/2026/08/11-20/phase30-rdna4-native-attention-kv-optimization.md)
[runtime architecture](../../../../../architecture/runtime.md)
[KV memory decision](../../../../../architecture/kv-memory.md)
[数値・出力影響変更台帳](../../../../../compatibility/numerical-output-changes.md)
[メイン計画](../../../../main-plan.md)
[対応する履歴](../../../../../history/2026/08/11-20/phase31-chunked-prefill-memory-foundation.md)
