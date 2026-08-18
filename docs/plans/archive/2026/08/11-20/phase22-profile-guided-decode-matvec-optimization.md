# Phase 22: profile-guided decode M=1 matvec optimization

> 状態: completed（P22-A4完了、candidate棄却、current v4維持）
> 作成日: 2026-08-18

## 完了結果

- 8 distinct shape、合計249 matvec/tokenを測る`phase22-matvec-profile-v1` evidenceを追加し、canonical V620/R9700で
  3 warmup + 10 measured、HIP-only、fallbackなし、all-one BF16 exact oracle、cleanup 0を確認した。current v4の
  calls-per-token加重medianはV620 39.344 ms、R9700 28.172 msだった。
- 最初のcandidateとしてRDNA wave32を1 outputへ割り当て、8 output/workgroupとするadditive variantを実装した。
  V620の`2560→9216`は約32%短縮したが、`9216→2560`は約35%悪化した。R9700の主要2 shape加重値も約13%悪化したため、
  R9700はoperator screeningで不採用とした。
- V620は改善した`2560→9216`だけへselectionを絞ってfull-modelを比較した。同一GPUにbaseline/candidateを同時常駐させ、
  順序を交互にした最終10組で、baseline/candidate中央値は254.668/255.982 msとなり、candidateが0.52%遅かった。
- 固定した受入条件11/12に従いcandidateを棄却し、新kernel ID/symbol/selectionをproduction sourceから除去した。
  全target/shapeは従来のdecode v4/wave64 providerを維持する。Phase 22を高速化成果とは表記しない。
- 独立f64 oracleはMLP down実shapeを加えた18ケースへ拡張し、Phase 21のserial-row provider identityも同期した。
  両GPU18/18、exact gfx1030/gfx1201/gfx942 release compile、host test、clippy、JSON/schema/H3 contractをPASSした。

## 目的

Phase 21で通常decodeのcompletion eventをsegment fenceへ集約する構造削減は成立したが、canonical V620/R9700の
end-to-end中央値は0.14%/0.18%遅く、wall改善を示さなかった。Phase 22は同じhost同期candidateを拡張せず、直近の
full-model profileでGPU時間の主因だったBF16 `M=1` matvec本体を、exact targetと実shapeに基づいて限定的に最適化する。

最初のwork unitはQwen3.5-4B BF16のdense text decodeに固定する。現行の単一MMVF providerと、`target/K/N`を使う
shape-aware candidateをoperator microbenchmarkとfull-model wallの両方で比較し、改善したshape/targetだけを採用する。
kernel、provider、数値順序を無制限に作り替えず、一度に一候補だけを実装・測定する。

## P22-A0で確認した開始時点の事実

- 現行`select_variant(m, k, n, target)`はBF16 decodeで`M == 1`とexact targetだけを使い、`K/N`を明示的に捨てている。
  RDNA2/RDNA4は`matmul.bf16_fp32.decode.v4`、gfx942はwave64 variantを選ぶ。
- RDNA2/RDNA4のdecode v4は256 thread、出力列ごとに1 workgroup、paired BF16 load、FP32 accumulation、BF16 RNE output、
  non-temporal weight loadを使う。K/Nやprojection roleによるworkgroup幅、rows-per-block、load幅の選択はない。
- Qwen3.5-4Bの1-token dense decode graphは32 layer（24 GDN、8 full attention）とfinal projectionからなり、
  BF16 `M=1` matvecは合計249回である。source graphから固定したrole/shape inventoryは次のとおり。

| role family | K | N | 1 token当たりの回数 |
| --- | ---: | ---: | ---: |
| MLP gate/up | 2,560 | 9,216 | 64 |
| MLP down | 9,216 | 2,560 | 32 |
| GDN/full q系 | 2,560 | 8,192 | 32 |
| GDN z | 2,560 | 4,096 | 24 |
| GDN/full out | 4,096 | 2,560 | 32 |
| full k/v | 2,560 | 1,024 | 16 |
| GDN b/a | 2,560 | 32 | 48 |
| tied vocabulary projection | 2,560 | 248,320 | 1 |

- Phase 15直前のbounded profileではdecode BF16 matvecがGemmaのdevice timeの84.28%、Qwenの63.28%を占めた。
  Phase 9ではQwenの1-token GPU時間約18.6 ms中MMVFが約15.5 msだった。これらは候補選定の根拠であり、Phase 22の
  fresh性能証拠として再利用しない。
- Phase 14→15 bridgeでdecode v4のstreaming weight loadはfull-model改善を示して採用済みである。Phase 22はv3へ戻さず、
  current v4をbaselineとする。
- Phase 21の結果からevent数だけを減らすcandidateのwall寄与は小さい。event pool、registry lock、graph replayを
  matvec candidateへ混ぜず、別work unitとして残す。

## Scope

### 対象

- Qwen3.5-4B BF16、単一request、batch 1、通常greedy text decodeの`M=1` matvec。
- exact `gfx1030` V620とexact `gfx1201` R9700。gfx942はsource/compile契約を回帰するが、実機採用claimは行わない。
- current v4に対するshape別kernel timing、full-model内のrole/shape別寄与、launch数、memory bandwidth指標のfresh profile。
- `target/K/N`で選ぶ小さなstatic provider tableと、既存数値契約を保つ1つのkernel candidate。
- 最初のcandidate familyとして、反復weight trafficが最大の`K=2560,N=9216`と`K=9216,N=2560`に対する
  workgroup幅、1 workgroup当たりの出力列数、paired/vector load、reduction構成の比較。
- `K=2560,N=248320` vocabulary projectionはA0/A1 profileへ含めるが、MLP shapeと同じvariantが有利と確認できない限り
  一つのcandidateへ暗黙統合しない。
- additiveなdispatch identity、evidence、oracle、host selection test、両GPU full-model比較。

### 非対象

- DeepSeek V4、TurboQuant、追加model family、追加model/KV形式。
- FP8/NVFP4/MXFP4、sparse MoE、MTP、visionのmatmul provider変更。
- gate/up+SiLU、RMSNorm producer、QKV、argmax等を跨ぐfusion。fresh profile後の別candidateとしてのみ扱う。
- token/position H2D統合、KV publication、event pool、registry lock、HIP Graph/native command-list。
- request/continuous batching、chunked prefill、prefix cache、永続化、multi-GPU、Infinity Fabric/RDMA。
- public CLI/API flag、GGUF/model-lock形式、README、release packaging。

## 固定する実行契約

1. matrix semanticsは`[M,K] x [K,N] -> [M,N]`、BF16 input/weight、FP32 multiply/accumulate、BF16 RNE outputとする。
   NaN/Inf classification、signed zero、odd/non-aligned tailを既存oracleより弱めない。
2. provider選択はprepare時にexact target、M/K/N、必要なら明示roleから決める。実行error後に別providerへretryせず、
   unsupported target/shapeは既存baselineを事前選択する。
3. current v4はrollback可能なbaselineとして維持する。candidateは新しいkernel ID/symbol/variantで識別し、evidenceで
   baselineと混同しない。
4. graphのsemantic op数、matmul submission数、tensor layout、weight bytes、state publication、completion policyを変更しない。
5. shape-aware tableは測定済みexact tupleだけをcandidateへ向ける。未測定shape、gfx942、M>1は既存providerを維持する。
6. offline/startup autotuning、runtime環境変数、ユーザー向けopt-inを最初のcandidateへ追加しない。static table採用後に
   複数variantが同程度で選択が不安定な場合だけ別work unitとして再検討する。
7. Phase 21のdeferred completionを性能laneへ混ぜず、PROFILED production baselineのまま比較する。

## 固定した受入条件

### A0/A1 profileと候補選択

1. current source、GGUF/derived lock、toolchain、exact target、clock/health、prompt/output、warmup/measured回数をmanifestへ固定する。
2. operator laneは上表の全distinct shapeをcurrent v4で測り、kernel median/MAD、effective bytes、target、kernel ID、fallback、
   cleanupを記録する。少なくとも主要MLP 2 shapeとvocabulary shapeは個別に扱う。
3. full-model profileはkernel symbol集計だけでなく、shapeまたはgraph roleへ帰属できるbounded instrumentationを用いる。
   instrumentation自体のwall値をproduction比較へ流用しない。
4. 最初に実装するvariantはfresh profileの最大寄与shape familyから一つだけ選ぶ。同率なら反復回数と両target共通性を優先し、
   追加fusionやhost/runtime変更へscopeを広げない。

### Correctness・dispatch

5. candidateはM=1の実shape、K/N非整列、Kとload/reduction境界の`B-1/B/B+1`、小さいN、odd K、zero、subnormal、
   finite大値、NaN/Infを独立FP32 oracleと比較する。powers of twoだけで合格にしない。
6. host selection testでexact target/M/K/Nごとのbaseline/candidate IDを照合し、unknown target、M>1、未採用shapeが
   baselineに留まることを確認する。
7. canonical V620/R9700でcandidate shapeのnative numerical testとQwen3.5-4B固定generationを実行する。
   token、stop、submission/kernel、segment/boundary、fallback、resident/peak、cleanupをbaselineと照合する。
8. CPU emulation/fallback、timeout、crash、zero case selection、別target binaryはGPU PASSにしない。

### 性能candidateの採用

9. operator draftは各targetでwarmup 3 + measured 10以上、baseline/candidateをcounterbalanceし、median、MAD、p10/p90、
   run-order driftを記録する。candidate shapeのkernel時間がcase固有noise envelopeを越えて改善しなければfull-modelへ進めない。
10. final full-modelはPhase 21と同じQwen3.5-4B BF16 audit GGUF、prompt `Hello world`、greedy 3-token outputを基準に、
    各targetでwarmup 3 + measured 10以上、交互順で比較する。より長いdecodeは改善方向のspotであり採用条件を置換しない。
11. candidateは対象shapeのoperator改善に加え、少なくとも一つのtargetでE2EまたはTPOTがcase固有noise envelopeを越えて改善し、
    他targetにnoise超過退行がない場合だけ該当target/shapeのdefaultへ採用する。target別採否を許す。
12. operatorだけ改善してfull-modelがnoise内ならproductionへ採用しない。棄却結果と残差を記録すればwork unitは閉じられるが、
    高速化済みとは表記しない。

### Integration・文書

13. affected Rust/native host test、exact `gfx1030`/`gfx1201`/`gfx942` compile、両GPU focused oracle/generation、format、clippy、
    diff/link/schema check、integration review 1回を行う。finding修正時はfindingだけをfocused再確認する。
14. llama.cppを追加adaptする場合は固定commit、bounded file/function、notice、import identityをprovenance記録へ追加する。
    現行v4の既存provenanceを新candidateの根拠として自動流用しない。
15. 採否後にruntime/compatibility/main plan/history/evidenceを同期し、Phase完了時に本planをarchiveへ移す。

## 実装・検証順序

### P22-A0: source inventoryとprofile contract（開始済み）

- current provider selector、decode v4 kernel、Qwen dense graphからtarget、shape、role、回数を棚卸しする。
- primary GGUF/lock、prompt/output、production PROFILED mode、canonical GPU visibilityを固定する。
- role/shape帰属のbounded profile手段とoperator evidence caseを決める。
- 最初の比較familyをMLP `2560→9216` / `9216→2560`に固定し、vocabulary projectionを独立controlにする。

### P22-A1: fresh baselineとvariant microdesign

- 両targetで全distinct decode shapeのv4 operator timingと、4B full-modelのrole/shape別device-timeを取得する。
- target別にworkgroup幅、outputs-per-workgroup、load/reduction構成を最大3 variantまで机上・microbenchmark比較する。
- 数値順序、occupancy、register/LDS、global weight traffic、launch構成を記録し、最大寄与familyの1 variantだけをA2へ進める。

### P22-A2: additive shape-aware provider candidate

- 新kernel ID/symbol/variantと`target/M/K/N` static selectionをadditiveに実装する。
- baseline選択、unknown/unaligned shape、M>1、gfx942を変更しない。
- host dispatch/fault testと境界数値oracleを先に通し、失敗時はfull-model測定へ進まない。

### P22-A3: dual-GPU correctnessとdraft performance

- exact gfx1030/gfx1201 buildとcandidate numerical matrixを実GPUでPASSさせる。
- Qwen3.5-4B固定generationのtoken/audit/memory/cleanupを照合する。
- operator counterbalanced比較でnoiseを越えるcandidateだけをfull-model final比較へ進める。

### P22-A4: final decisionとcloseout

- 両targetでcounterbalanced full-model比較を行い、target/shape単位で採用または棄却する。
- integration review 1回、findingのfocused re-review、compatibility/evidence/main plan/history同期を行う。
- 次candidateを同Phaseへ追加するのは最初のwork unit完了後、fresh profileで最大寄与が明確な場合だけとし、
  一度に一候補の制約を維持する。

## 計測lane

| lane | 用途 |
| --- | --- |
| M0 | host selector、kernel identity、境界oracle、fault/cleanup。GPU性能claimに使わない |
| M1 | exact GPU operator microbenchmark、全distinct shape、3 warmup + 10 measured以上。candidate screening |
| M2 | instrumented 4B decode、shape/role帰属。profile overhead込みwallを採用根拠にしない |
| M3 | production 4B固定lane、両GPU、3 warmup + 10 measured以上、counterbalanced。最終採否 |

## Rollback・停止・再計画

- 数値順序、token、dispatch accounting、fallback、cleanupが一致しないcandidateは性能に関係なく棄却する。
- fresh profileでM=1 BF16 matvecが支配的でない場合はkernel実装を開始せず、最大寄与項目へPhase 22を再計画する。
- operator改善がnoise内、またはfull-modelがnoise内/退行ならcurrent v4を維持し、variantをdefaultへ残さない。
- 最初のcandidateへfusion、quantization、graph replay、batchingを追加して救済しない。それぞれ独立候補として再計画する。
- 同じwork unitの2回reject、review時間が実装時間超、1時間以上の機能進捗停止、検証/docs 30%超、見積り1.5倍超、
  acceptance変更時は追加探索・検証を止めて同じwork unitを再計画する。

[対応する履歴](../../../../../history/2026/08/11-20/phase22-profile-guided-decode-matvec-optimization.md)
