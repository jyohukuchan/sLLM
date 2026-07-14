# AQ4 production prefill/decode optimization plan v0.1

Status: P0 complete; P1 schema/mechanics gate complete; P1-D and real production performance are handed off to P2; active product remains unchanged until P6 unless the explicit P4 prefill-only exception is approved

## 前回の要点

- Qwen3.5-9B AQ4_0 reasoning v2は、R9700上のresident worker、Gateway、OpenWebUI、HTTP/SSE、停止・障害復旧、100-chat soakまでrelease gateを通過している。
- production decodeはtyped backend registryとpaged split pathへ接続済みで、context 1339から64 tokenを生成する測定はprefill約129.33 tok/s、decode約66.53 tok/sだった。`cache_len < 256`と対応外条件はcanonical pathへfail-closedする。
- production prefillは約117–129 tok/sであり、`docs/specs/prefill-validation-v0.1.md`がQwen3.5 AQ4/R9700向けに定めるprompt 1011の最低318.19 tok/s、prompt 2048の旧tokenwise比5倍、prompt 1024の目標1000 tok/sには達していない。
- adapter fixtureとtyped executable registryの一部は実装済みだが、fixtureは完全なproduction package adapterではない。P1のtrace producer/validatorはCPU mechanics fixtureを通過するが、GenericModelExecutorとgeneric M>1 prefillのresident worker接続はP2 handoffである。
- 最終release bundleに記録されたrollback environment hashと現在の`/etc/ullm/openai-gateway-manifest.env`のhashが異なる。次のactivationやrollbackより前に理由を確認し、現状態へbindingし直す必要がある。

## 今回の変更点

- prefill/decode最適化へ入る前の必須作業を、identity/rollback、production trace、独立検証、固定baselineの四点に限定した。
- 実装をprefillとdecodeに分け、P2のbaseline取得後はprefill実装とdecodeの読み取り中心の解析を並列に進められる構成にした。
- CPU oracle、trace validator、benchmark/evidence、kernel実装をファイル所有権の異なるlaneに分けた。共有ABI、session、registry、R9700測定、activationは直列の統合窓口とする。
- componentの改善をproduction性能として扱わず、component、full model、direct worker、API/SSE、OpenWebUIの順に同一identityを追跡するgateを置いた。
- OOM、予期しないfallback、状態不一致、短文脈decode回帰、identity不一致を候補停止条件として明文化した。

## 次の行動

P0とP1 schema/mechanicsの基盤実装とcurrent identityの固定を完了した。次はP2で現active identityのbaselineを取得し、同一identityで比較可能なprefill/decode測定へ進む。P1 CPU fixtureは独立validatorへbinding済みだが、direct workerは`full_model`であり、`production_server`の実boundary証拠はP2で再取得する。

### 2026-07-14 実装状況

- P0: active manifest、worker、package、tokenizer、Git、systemd、Gateway、worker、OpenWebUI、GPU/driver/power conditionの非秘密snapshotと、現在のmanifest-mode environmentへ再bindingしたrollback artifactを保存した。旧bundleのenvironment hashとの差はlegacy environmentからmanifest-mode environmentへのsystemd drop-in切替として説明した。
- P1-A/B: engine-side bounded trace primitive、production executor record契約、strict trace producer、duplicate/privacy/path/hash/算術を再計算する独立validatorを追加した。wire protocolは変更していない。CPU fixture `tests/fixtures/production-execution-trace-p1/schema-r1/` は detached validator report SHAを含む独立検証を通過する。
- P1-C: 2 warmup + 10 measuredのatomic matrix runnerとappend-only evidence validatorを追加した。runnerの既定`mechanics_smoke`は性能証拠ではなく、P2の実commandは`--mode production`で明示する。
- P1-D: wall time・launch・workspace・fallback・実Mのread-only auditを診断専用として追加した。P2で実際のproduction-server traceから再監査するまで、候補昇格や性能比較には使わない。
- 既存のlive evidenceは保持する。旧traceのproduction claimはschema-r1へ再検証されるまでpromotion根拠にしない。サービスの再起動・設定変更・active manifestの変更は行っていない。

## 1. 目的と適用範囲

この計画は、Qwen3.5-9B AQ4_0をRadeon AI PRO R9700で動かす現行production pathのprefillとdecodeを最適化するための実行計画である。

上位計画は`docs/plans/generic-production-inference-optimization-plan-v0.1.md`、合格契約は次の仕様とする。

- `docs/specs/prefill-validation-v0.1.md`
- `docs/specs/production-execution-trace-v0.1.md`
- `docs/specs/inference-benchmark-result-v0.1.md`
- 現行AQ4 reasoning release仕様とrelease evidence

本計画は上位計画を置き換えない。完全なgeneric executorへの全モデル移行を待たず、現行AQ4 product pathに必要なtraceとidentity bridgeを先に完成させる。ただし、新しい実装をモデル名分岐としてsessionへ直書きせず、typed graph、state schema、backend registry、workspace admission、fallback chainの境界を守る。

### 1.1 成功条件

最終成功条件は次のすべてである。

1. production_server scopeのtraceが、実際のchunk幅、implementation ID、fallback、workspace、observed peak、state commit/reset、binary/manifest/package/graph identityを記録する。
2. all-M=1 path oracleと独立source oracleに対して、token、logit/hidden、KV、recurrent/conv state、position、chunk境界、cancel/resetが合格する。
3. prefillは仕様のQwen3.5 AQ4/R9700向け最低条件を満たす。prompt 1024の1000 tok/sは目標として別に報告する。
4. context 1339のdecodeは、少なくとも現active identityのbaselineへ回帰せず、短文脈decode p50は5%を超えて回帰しない。
5. direct workerからOpenWebUIまで同じcandidate identityを追跡でき、release validator、browser、停止・障害復旧、soak、rollbackが合格する。
6. 新しいOOM、予期しないfallback、未完了のstate reset、identityの混在がない。

### 1.2 今回は行わないこと

- 複数request batching、request queue、prefix cache共有
- context limit 4096の拡大
- TLS、multi-tenant auth、Responses API、画像入力
- SQ8の再最適化またはSQ8のproduction移行。validation仕様が要求するSQ8_0のcross-format controlは実行する
- すべてのモデルをgeneric executorへ移すためだけの大規模refactor
- active serviceの早期activation

これらはprefill/decodeの測定とpromotion gateを不必要に広げるため、別計画とする。

## 2. 現在の固定基準

計画作成時点の観測値とidentityは、P0で改めてhash-bound artifactへ固定する。次の値は方向付けのための現在値であり、候補比較にはP2で同一環境から取り直したbaselineを使う。

| 項目 | 現在値または状態 |
|---|---|
| active model | Qwen3.5-9B AQ4_0 reasoning v2 |
| active manifest SHA-256 | `feb3190d0ff59778e4da140b8db2bd1ce2ba440e3a69e844b997011d4d08cb44` |
| active worker SHA-256 | `177f3106414efc7cc4b08fa2d87bed6e147d4188e0a290f43b7a1ac591fae48d` |
| historical prefill | 約117–129 tok/s。旧manifest/source identityのp1339/g64で129.327636 tok/s。active baselineではない |
| historical long decode | 旧manifest/source identityのp1339/g64で66.526692 tok/s。active baselineではない |
| decode selection | `cache_len >= 256`でpaged split、短文脈と対応外条件はcanonical fallback |
| product concurrency | active GPU request 1件、queue/batchingなし |
| context/input | 4096、text-only Chat Completions |

表の性能値を含む過去のBM8、paged split、tiled GEMM、wave-softmax等の結果は探索資料として使えるが、source、worker、manifestのidentityが現activeと異なる証跡をpromotion根拠へ流用しない。現active identityのbaselineはP2で新規取得する。BM8は現identityで再測定する。回帰したtiled GEMMとwave-softmaxは、設計上の変更と新しい仮説がない限り再試行しない。

## 3. 依存関係とcritical path

```text
P0 identity/rollback
  -> P1 trace + validator + benchmark foundation
  -> P2 frozen baseline/profile
  -> P3 prefill candidate
  -> P4 prefill freeze
  -> P5 decode candidate
  -> P6 combined production promotion
  -> P7 closeout
```

P2完了後、P5のうちread-only decode解析はP3と並列に開始してよい。GPU計測と共有ファイル編集は並列化しない。prefillだけを先行activationする場合は、P4の明示的な判断点でdecode非回帰を証明し、別のdecode計画をbindingする。

## 4. Phase計画

### P0: identity、rollback、安全条件の固定

並列性: 原則直列。読み取り監査だけは並列可。

作業:

- Git commit、worktree、binary、worker、package、tokenizer、active manifest、driver、GPU、power conditionをcaptureする。
- release bundleのrollback environment hashと現environment fileの差分を内容単位で説明する。秘密値は記録しない。
- `ullm-openai.service`、AQ4 worker process、Gateway、OpenWebUIの実際のtopologyを確認する。unit名だけから異常と断定しない。
- 現状態にbindingしたrollback artifactを再生成し、独立validatorで検証する。active manifestとserviceは変更しない。
- run root、case naming、hash-bound threshold policy、R9700測定queueを固定する。
- OpenWebUI managed model markerと明示的`thinking_budget_tokens`経路の未reconcile状態を記録し、P6の必須確認へ入れる。

成果物:

- immutable environment/identity snapshot
- current rollback bindingと検証結果
- case matrix、threshold policy、GPU queue
- P0 decision record

Gate:

- 現active状態を同定でき、rollback artifactのhash差分が説明または解消されている。
- active製品を変更していない。
- baselineとcandidateが同一identity条件で比較できる。

### P1: 測定・trace・独立検証の基盤

並列性: A、B、C、Dを並列可。session/workerへの統合は直列。

#### P1-A: production execution trace producer

- 現行AQ4 sessionから、`ullm.production_execution_trace.v1`に必要なgraph/state、resolved operator、実chunk幅、fallback、workspace、peak memory、state transition、binary/manifest/package identityを生成する。
- traceは4 MiB以下に制限し、prompt本文、response本文、API key、OpenWebUI DB内容を含めない。
- component、full_model、production_server scopeを混同しない。

主な所有範囲:

- 新規のengine trace module
- trace専用test/fixture
- production trace schemaに必要な最小限の追加文書

#### P1-B: 独立validator

- producerの`passed`を信用せず、duplicate key、non-finite、symlink、path escape、hash、identity、case completeness、fallback、state resetを再計算する。
- trace/result/policy/oracle/binary/manifest/packageをhashで結ぶ。
- validatorがproducer summaryからではなく実行事実を再構成できるよう、phase実幅、operator選択とcompatibility、workspace、memory算術、prepare/commit/discard/resetを記録するbounded sanitized executor-record sidecarを定義する。sidecarはtraceからSHA-256でbindingする。
- planned tool: `tools/validate-production-execution-trace.py`

#### P1-C: benchmark/evidence runner

- all-M=1、cold batched、cached-prefix chunked、decodeを同一case schemaで実行する。
- 2 warmup + 10 measuredを既定とし、p50/p95、TTFT、ITL、VRAM peak、fallback countを保存する。
- `.incomplete`からatomic renameし、既存evidenceを上書きしない。
- planned tools: `tools/run-aq4-production-performance-matrix.py`、`tools/validate-aq4-production-optimization.py`

#### P1-D: read-only bottleneck audit

- kernel timeline、launch数、D2H/H2D、stream synchronization、workspace allocation、fallback、実Mを調べる。
- paged KV block-table validation、scale metadata residency、AQ4 projection、linear/recurrent attention、dense self-attention、embedding/LM headを候補として順位付けする。
- 最適化案はwall timeへの寄与とproduction traceで選び、名前や過去の期待値で選ばない。

統合Gate:

- CPU fixtureとproduction smokeのtraceが独立validatorを通る。
- full production graph identityがembeddingからLM headまで追跡できる。fixtureだけをproduction adapterと称さない。
- unexpected fallback、OOM、incomplete resetをvalidatorがfail-closedで拒否する。
- traceがない候補はdiagnostic測定までに留め、promotionしない。

### P2: baseline凍結とprofile

並列性: raw evidenceの解析とvalidator実行は並列可。R9700を使う実行は1本ずつ直列。

作業:

- 現active sourceからclean baseline binaryを作り、all-M=1と現production pathを同じrun rootへ記録する。
- 最初に代表点をprofileする。全matrixをprofileせず、通常測定は全case、詳細profileはボトルネックを区別できるcaseへ絞る。
- prefill代表点: prompt 128、512、1011、1024、1339、2048、3584。
- decode代表点: start context 16、128、512、1024、1339、2048、3584、原則64生成token。
- M grid: 1、8、16、32、64、128。resolved Mとfallbackを必ず記録する。
- cold prefillと、対応する場合はcached-prefix chunkedを測る。未対応機能を擬似的に成功扱いしない。
- source oracle、path oracle、state snapshotをstreaming/chunk単位で比較し、全logit matrixをメモリへ保持しない。

成果物:

- `benchmarks/results/YYYY-MM-DD/qwen35-9b-aq4-production-opt-v0.1/`
- immutable baseline JSONL、trace、oracle、profile、SHA256SUMS
- bounded sanitized executor-record sidecarとtraceからのhash binding
- wall time、launch/sync、transfer、workspace、fallback別のranked bottleneck report

Gate:

- baselineのばらつきとp50/p95が説明可能である。
- current activeとのidentity差がないか、差がある場合は比較不能として分離されている。
- optimizerが最初に扱う一つのbottleneck familyを選べる。

### P3: prefill候補の実装と選抜

並列性: ABI/descriptor凍結後は、runtime kernel、CPU oracle、evidence toolingを別laneで並列可。registry/session統合とGPU実行は直列。

候補の優先順はP2のprofileで決める。想定候補は次である。

1. paged KV block-table validation等に伴うD2Hとstream synchronizationの削減
2. AQ4 BM8/register kernelのshape coverage、tail、scale metadata residency
3. projection、norm、activation、residualのlaunch削減または安全なfusion
4. recurrent attentionとdense self-attentionのchunk execution
5. embeddingまたはLM headが実測上支配的な場合の専用改善

Lane:

| Lane | 責務 | 所有範囲 | 統合条件 |
|---|---|---|---|
| P3-A runtime ABI/kernel | HIPRTC source、workspace、kernel launch | `runtime/src/ullm_runtime_api_aq4.inc`、`runtime/src/ullm_runtime_hiprtc_sources.inc`と専用test。共有header/APIはこのlaneだけが編集 | descriptorとABIを先に凍結 |
| P3-B CPU oracle/state | M grid、chunk境界、KV/recurrent/conv transaction比較 | `cpu_reference_executor.rs`と新規oracle test。production traceは生成しない | source/path oracleを先に固定 |
| P3-C registry/engine integration | capability、workspace admission、priority、fallback、session接続 | `backend_operation_registry.rs`、AQ4 runtime/session関連。共有ファイルは統合担当だけが編集 | A/B合格後に直列統合 |
| P3-D evidence | case生成、raw capture、独立validation | `tools/`の新規runner/validator、fixture | 実装laneのsummaryを信用せずrawから判定 |

各候補は次の順で上げる。

```text
CPU/source differential
  -> HIP component M grid
  -> full model offline
  -> direct resident worker
  -> production_server smoke
```

候補選抜Gate:

- shape、dtype、finite、hidden/logit、greedy token、top-k、KV/recurrent/conv/cache/position、chunk境界が合格する。
- cancel、publish failure、EOS、length、reset、次requestがbaselineと同じ状態へ戻る。
- 新しいOOM、予期しないfallback、workspace policy超過がない。
- すべてのcase/resultは失敗も含めてimmutable evidenceとして保存する。prefill p50が5%超、p95が10%超回帰するcandidateはpromotionから外す。
- full-modelで測定誤差を超える改善がない候補はproduction統合しない。componentだけの改善を採用理由にしない。
- 同時に育てるcandidate familyは原則二つまでとし、R9700 queueを分散させない。

### P4: prefill freezeとpromotion判断

並列性: 判定は直列。文書化とarchive checksum計算は並列可。

Gate:

- prompt 1011のproduction prefillが318.19 tok/s以上である。
- prompt 2048がhash-bound policyの旧tokenwise baseline比5倍以上で、OOMがない。
- prompt 1024の1000 tok/s達成可否を、必須条件と混同せず報告する。
- context 1339 decodeが現active baselineから5%を超えて回帰しない。
- short-context decode p50が5%を超えて回帰しない。
- production_server traceとindependent validationが合格する。

既定ではここでactivationせず、prefill candidateをfreezeしてP5へ進む。prefill-only activationは君が明示的に選択した場合だけ行い、rollback-ready bundleと独立したdecode計画を必須とする。

### P5: decode解析、候補実装、選抜

並列性: P2後のread-only解析はP3と並列可。コード変更はP3とファイルが重ならない期間または別worktreeで行い、registry/session統合とR9700測定は直列。

保持するもの:

- production昇格済みのpaged decode split
- `cache_len < 256`のcanonical path
- feature/workspace/capability不足時のfail-closed fallback
- caller-owned persistent workspaceとstate commit/reset契約

調査対象:

- 残存AQ4 matvec/projection
- linear/recurrent attention
- dense self-attentionの長文脈scan
- LM head/top-1
- host synchronizationと小kernel launch
- workspace再利用、state read/write、block-table validation

Gate:

- context 1339で仕様のhistorical floor 53.3 tok/sを満たし、さらに同一identityの現active baselineと比較して回帰しない。
- context 16、128、512のdecode p50が5%を超えて回帰しない。
- exact greedy token、state、cancel/reset、publish failure、次requestが合格する。
- profiled timelineと通常throughputの両方で改善を確認する。profileだけ速い候補を採用しない。
- full-modelで測定誤差を超える改善がない候補はproduction統合しない。

### P6: 統合candidateのproduction promotion

並列性: evidence assembly、validator、文書確認は分離可。build identity固定、GPU run、activation、rollback rehearsalは直列。

順序:

1. clean candidate buildとartifact hash固定
2. component validation
3. full-model offline validation
4. direct worker non-stream/SSE
5. Gateway API/SSEと全reasoning mode
6. OpenWebUI browser、Stop、worker failure/recovery
7. 通常100-chat/resource soak、restart後20-chat recovery、restart count確認
8. release bundle、promotion receipt、rollback bindingの独立検証
9. canary activation
10. post-activation probeまたは即時rollback

Gate:

- prefill P4とdecode P5の全Gateを同じcandidate identityで満たす。
- active manifest、unit、environment、worker、binary、package、traceがrelease bundleと一致する。
- OpenWebUI managed model markerと明示的`thinking_budget_tokens` requestが現active identityへbindingされる。
- HTTP/SSE、reasoning hidden-content非再投入、Stop、EOS/length、failure recovery、restart count、VRAMが合格する。
- model ID/name overrideを現manifestへbindingし、既存OpenWebUI imageを使い、UI patchを新たに加えない。
- busy時のdirect API 429と`Retry-After`契約を回帰させない。OpenWebUI側の既知のvisible 400変換は別途記録する。
- rollback rehearsalが現在のenvironment hashで再現できる。

### P7: closeout

並列性: evidence archive、文書、backlog更新を並列可。最終commitは統合担当が直列に行う。

- raw evidence、validator output、SHA256SUMS、release bundleをimmutable archiveへ確定する。
- 本計画と上位generic planのstatusを実装実態に合わせて更新する。
- 採用/不採用candidate、理由、回帰、OOMをjournalへ残す。
- generic executorの残作業、request batching、prefix reuse、SQ8移行を別backlogへ切り出す。
- legacy pathはrelease gate完了まで削除せず、oracle/rollback用途を明記する。

## 5. 並列実行表

| Phase | 並列可 | 直列必須 |
|---|---|---|
| P0 | read-only identity/service監査 | rollback binding、policy freeze |
| P1 | trace producer、validator、runner、read-only audit | session/worker統合、schema freeze |
| P2 | evidence parse、validation、report | R9700実行、実モデルGPU runtime test |
| P3 | kernel、CPU oracle、evidence tool。ABI凍結後のみ | shared ABI、registry/session統合、R9700実行 |
| P4 | archive/checksum、文書 | candidate freeze、promotion判断 |
| P5 | decode read-only解析はP3と並列可 | shared registry/session編集、R9700実行 |
| P6 | evidence assembly、独立validator、文書確認 | clean build、GPU run、activation、rollback |
| P7 | archive、文書、backlog | status/commit統合 |

## 6. 共有resourceと競合回避

### 6.1 R9700 lock

- R9700でGPU processを起動する測定は常に一件とする。
- resident worker、実モデルCLI、workspace GPU test、llama.cpp/vLLM比較を同時実行しない。
- 実package GPU smokeとworkspace testは、過去に並行時だけsegfaultが観測されたため必ず逐次実行する。
- baselineとcandidateは同じmanifest、binary、model、driver、power condition、warmup/repeatで同じ測定窓に並べる。
- device selectionは現在の構成をP0で再確認し、`ROCR_VISIBLE_DEVICES`を優先して記録する。固定indexを確認せず使わない。
- OOMは成功結果で上書きせず、immutableな失敗artifactとして保存する。

### 6.2 shared file lock

次は統合担当だけが直列編集する。

- `crates/ullm-engine/src/lib.rs`
- `crates/ullm-engine/src/backend_operation_registry.rs`
- `crates/ullm-engine/src/qwen35_aq4_session.rs`
- runtimeのpublic header、共有ABI、共通entrypoint
- worker protocol/driverの共有層
- served-model manifest、systemd、release bundle

並列candidateは別Git worktreeを使い、同じファイルを同時編集しない。各laneは新規ファイルまたは専有ファイルでtestを完成させ、統合窓で一つずつ取り込む。

## 7. 必須validation matrix

仕様の全matrixを最終promotionで満たす。開発中は段階的に広げる。本計画で変更する実装はQwen3.5 AQ4だけだが、Qwen3 dense/SQ8_0/referenceのfull-model control evidenceも上位validation dependencyとしてhash-boundで再利用または再取得する。現在のpolicyとidentityへbindingできる証跡がなければP6 promotionを止める。

| 軸 | 必須値 |
|---|---|
| topology | Qwen3 dense full-model control、Qwen3.5 hybrid production target。fixture/componentだけでは代替不可 |
| format | AQ4_0 target、SQ8_0 cross-format control、利用可能なBF16/FP16/F32 reference/source oracle |
| prefill M | 1、8、16、32、64、128 |
| prefill mode | all-M=1、cold batched、cached-prefix chunked。unsupportedはimmutable evidenceとして記録するが、明示的に適用外とするapproved policyがない限りpromotion coverageを満たさない |
| prompt/context | 1、8、32、128、512、1011、1024、1339、2048、3584、context edge |
| decode start | 16、128、512、1024、1339、2048、3584 |
| scope | component、full_model、production_server |
| backend | CPU reference、R9700 mandatory、V620 capability smoke/support decision |
| failure | invalid shape、workspace不足、OOM、cancel、publish failure、worker failure、reset、rollback |

V620は補助的なcapability確認であり、R9700のacceptance authorityを置き換えない。R9700通過前に大規模V620測定を並行させない。

## 8. 停止・rollback条件

次のいずれかでcandidateを停止し、production selectionから外す。

- source/path oracle、token、state、chunk境界が不一致
- 新しいOOM、VRAM/workspace policy超過
- undeclared/予期しないfallback、resolved batch幅不足
- prefill p50が5%超、p95が10%超回帰
- short-context decode p50が5%超回帰
- full-modelまたはproduction_serverで改善が消える
- binary、manifest、package、trace、policyのidentity不一致
- cancel、publish failure、EOS/length、reset、次requestの状態不一致
- active serviceの安定性、restart count、OpenWebUI動作の回帰

activation後に上記が発生した場合は、追加測定を続けず、P0/P6で検証済みのbundle-bound rollbackを実行する。

## 9. 作業記録

各Phaseで次を残す。

- `journal/YYYY/MM/DD/<work>.md`: 仮説、変更、測定、採否、次の行動
- run root: raw evidence、trace、case、policy、oracle、validator output、SHA256SUMS
- Git: 意味のある実装単位のcommit。evidence identityをcommitへbindingする

prompt本文、response本文、prompt/generated token ID、API key、password、OpenWebUI DBの個人情報はtrace、release evidence、journalへ保存しない。公開fixture ID、hash、集計値だけを残す。局所診断でtoken IDが不可欠な場合もpromotion run rootとrelease bundleの外に隔離し、traceやbundleから参照しない。
