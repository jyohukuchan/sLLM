# AQ4 Phase 3b: production/harness configuration audit v0.1

## 前回の要点

- Phase 1 は、既存の `runtime_host_linear_attn_recurrent_f32` を使うCPU layer 0診断で、BF16 sourceに対する相対L2 `0.042451` を得た。AQ4のlayer-localな量子化誤差としては妥当な範囲だった。
- Phase 2/2c はlayer 0--11をCPUでchainしても、誤差が単純には深さ方向へ増えず、既知の最終logits相対L2 `0.6151289249` をH8（単純な深さ方向の蓄積）だけでは説明できないと判定した。Phase 2bではpost-norm epsilon差も無視できる規模だった。
- したがって本Phase 3bでは、数式を再検証せず、CPU診断harnessとresident実行経路の構成差（H6）をコードと既存evidenceから監査する。GPU実行、service/systemd、active manifest、07/16に停止したP3 harnessには触れない。

## 今回の変更点

### 結論

- **H6は、07/14に記録された既知の最初の不一致（`decoder_layer:0`）およびその根となるM=1差分を説明できない。** cold/warm、request間KV再利用、M>1 chunk、RoPE、paged KVのいずれも、当該M=1/cold traceのlayer 0より前には入らない、またはlayer 0で既に不一致がある。
- ただしH6を一般のonline servingについて完全否定したわけではない。通常Gatewayのprefillは2--128 tokenのnative sequence pathを取り得るため、そのM>1経路が後段に追加誤差を作る可能性は未測定である。
- CPU standalone/referenceとHIPのM=1 fused kernel・f32 reduction orderの差が、残る最も直接的な候補である。従って**H5（GPU kernel固有）を有力に格上げ**する。ただしGPU実行比較はPhase 3cの範囲であり、本監査だけでH5を証明したものではない。

### まず訂正すべきevidenceの位置付け

07/14の最終相対L2 `0.6151289249` は、OpenAI Gatewayへの推論requestではなく、同じproduction packageを直接loadした `ullm-aq4-p2-path-oracle` のGPU診断から得られた。`path-oracle-gpu-run-v1/command.json` はbinaryを `target/debug/ullm-aq4-p2-path-oracle`、`service_was_stopped=true` と記録し、対応journalも `--prefill-m 1` の実行であることを記録している（`journal/2026/07/14/qwen35-aq4-p2-path-oracle.md:64-90`）。

attempt3もGateway requestではない。artifactは `mode=aq4_gpu_intermediate_diagnostic`、runtime sidecarは `mode=diagnostic_only` / `model_loads=1` / `rows=3` である。これはproduction workerを変更せず同じresident session/model runtimeを専用 `ullm-aq4-differential-trace` binaryから直接駆動して、path-oracleの最初の不一致位置を観測したものだった。従って07/15 journalの「production run」はproduction packageを対象にした管理済みGPU診断という意味であり、Gatewayの実requestを意味しない。

attempt3 analysisは全3 rowでembedding後の最初の不一致をlayer 0としている。

| case / step | context length | first mismatch | sampled max abs |
| --- | ---: | --- | ---: |
| `fixture-prompt-0` / 0 | 3 | `decoder_layer:0` | 0.0054091103 |
| `fixture-prompt-0` / 1 | 4 | `decoder_layer:0` | 0.0048871059 |
| `fixture-prompt-1` / 0 | 2 | `decoder_layer:0` | 0.0024267659 |

Evidence: `benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/differential-trace-gpu-v1-attempt3/{manifest.json,runtime.json,payload.jsonl}` and `differential-trace-gpu-v1-attempt3-analysis/analysis.json`.

### 実際のonline Gateway resident path

通常のOpenAI Gateway requestは、以下の経路でlayer 0以降へ到達する。ここでのMは同時request数ではなく、1 requestのprefill token-sequence widthである。Gateway/workerは1 active requestを守るため、request batchingによる別requestのstate混合はこの実装にはない。

```text
POST /v1/chat/completions
  -> app.py: promptをtoken IDsへ正規化
  -> WorkerSupervisor: resident workerへJSONL `generate`
  -> ullm-aq4-worker: resident modelを1回loadしSessionInferenceBackendを作成
  -> drive_worker_request
       start_request
       -> prepare_advanceを反復
            prefill: M=1ならdispatch_token_for_phase
                     M=2..128ならdispatch_prefill_chunk_for_phase
            decode: M=1のdispatch_token_for_phase
       -> finish_and_reset（cancel時はabort_and_reset）
  -> Qwen35Aq4ModelRuntime::dispatch_layer_stack
       embedding -> decoder layer 0 -> ... -> decoder layer 31 -> final norm / LM head
```

根拠となる主な入口は、`services/openai-gateway/src/ullm_openai_gateway/app.py:238-332`、`worker.py:292-414`、`crates/ullm-engine/src/bin/ullm-aq4-worker.rs:403-461`、`session_worker_backend.rs:39-107`、`worker_driver.rs:190-291`である。

- workerの通常session configはgreedy prefill chunk上限128であり、sessionのM gridは`[1, 8, 16, 32, 64, 128]`である（`qwen35_aq4_session.rs:28-34,395-405`）。短いpromptやtailでは物理Mは1にもなる。
- `start_request`はReady stateだけを受け入れ、`prompt_tokens_processed=0`、`generated_tokens=0`、`decode_input=None`で開始する。最初のchunkだけが`ColdPrefill`、prompt内の後続chunkは`CachedPrefixPrefill`である（`qwen35_aq4_session.rs:1991-2025,2456-2640`）。decode位置は`prompt_len + generated_tokens - 1`となる。
- 終端時の`reset_all_request_state_synchronized`は全layerのrequest-owned stateをclearする。linear-attention layerではconv history/recurrent state、self-attention layerではK/V cacheと`written_len`をzero化する（`qwen35_aq4_model_runtime.rs:1403-1416`; `qwen35_aq4_layer_runtime.rs:2367-2396,4858-4887`）。residentなのはweightsであり、成功・cancelをまたぐwarm KV/recurrent stateではない。
- KV block sizeは256である（`qwen35_aq4_model_runtime.rs:420`）。現在のmodel runtimeは固定identity block tableを保持し、requestごとに別利用者へページを割り当てるscheduler/allocatorではない。paged KVはself-attention用のintra-request stateである。

P2の `ullm-aq4-p2-resident-driver` はこのGateway workerとは別のbenchmark/direct driverである。同じmodel runtimeを使うが、online request経路の証拠として混同しない。なおこのdirect driverはoffset>0 chunkにも`ColdPrefill`を渡す箇所があり、通常sessionの`CachedPrefixPrefill`とは異なる。これは複数chunkのdirect P2 benchmarkには影響し得るが、M=1のpath-oracle/attempt3にもGateway requestにも到達していない。

### 07/14 path-oracle / attempt3が実際に通った経路

- path-oracleはdirect binaryで`--prefill-m 1`を指定した。従って最終`0.6151289249`の比較はM>1 native prefillではない。
- attempt3のfrozen source（commit `28ec343a`）は`Qwen35Aq4SessionConfig::with_prefill_chunk_tokens(1)`を明示する（`ullm-aq4-differential-trace.rs:645-649`）。`token_ids.len()==1`は`dispatch_token_for_phase`に入る（当時の`qwen35_aq4_session.rs:565-598`）。
- caseごとに`start_calibration_request`し、terminal後に`finish_and_reset()`する（`ullm-aq4-differential-trace.rs:675-746`）。`fixture-prompt-0` step 0は`[11,12,13]`のcold prefill、step 1は同一request内でsource replay token `220`をcommitした`[11,12,13,220]`、`fixture-prompt-1`はreset後の`[21,22]`である。
- step 1の継続はAQ4予測tokenではなくsource tokenを強制replayする。この制御差は後続rowだけに影響し得るが、step 0とreset後の別caseでもlayer 0不一致がすでにあるため、最初の不一致の説明にはならない。
- attempt3はmodel runtimeのKV block 256構成を省略していない。しかしlayer 0はlinear attentionであり、RoPE/cache position/paged K/Vを使うself-attentionより前である。`dispatch_layer_stack`がlayer 0へ渡すのはembedding出力で、layer 0の不一致をpaged KVやRoPEで説明することはできない（`qwen35_aq4_model_runtime.rs:973-1018,1446-1522`）。

### harnessとの一致点・相違点と数値への評価

| 項目 | CPU harness | online Gateway | 07/14 path-oracle / attempt3 | 数値への評価 |
| --- | --- | --- | --- | --- |
| request初期状態 | contextごとにcoldのconv/recurrent stateからfull replay | request終端ごとに同期reset | path-oracle/attempt3ともM=1。attempt3はcase終端でreset | request間warm stateは既知layer 0差の原因にならない。step 1の同一request継続はfull replayと同じtoken列を表す。 |
| token位置・context | host chainは先頭からtoken順にpositionを進める | absolute position、後続prompt chunkはcached-prefix phase | cold M=1で先頭から逐次実行 | RoPE/positionはself-attention側でありlayer 0のfirst mismatchには非因果。 |
| paged KV | chainのself-attentionはhost Vec上のcausal attentionで、device paged cacheを通らない | block 256のdevice K/V cacheをread/write | 同じmodel runtimeのKV構成をload | 後段の追加差にはなり得るが、linear-attention layer 0の最初の差を作れない。 |
| M/chunk | 常にtoken-by-token（M=1） | prefillは実効M=1..128、M>1はnative sequence kernel | path-oracle/attempt3はM=1固定 | online M>1は未検証の追加リスク。ただしM=1で既にfirst mismatchがあるため、既知のroot mismatchを説明しない。 |
| request concurrency | なし | workerはsingle active request | direct diagnostic | request batch/KV reuseによるstate混合は見つからなかった。 |
| AQ4 payload / dequant入力 | `load_single_diagnostic`が同一packageをCPU contextへload | resident layerが同じpackageをHIP contextへload | GPU resident M=1 layer | CPU専用のpayload/scale簡略化は確認できなかった。`load_single_diagnostic`が省くのはbatch-plan admissionであり、weight/scale materializationは共有される。 |
| matvec・fusion・reduction | 各projectionを個別M=1 `.matvec()`、host recurrent | M=1でもQKV/Z/A/B/gate/beta fused、out `matvec_add`、fused MLP | 上記のGPU M=1 fused path | 最も重要な相違。後述のH5境界であり、CPU harnessだけでは数値等価性を証明できない。 |

CPU harnessは`RuntimeContext::create(0)`後にCPU backendを強制し（`crates/ullm-engine/src/bin/ullm-aq4-layer0-family-isolation.rs:831-841,1680-1698`）、`PackageAq4ResidentMatvec::load_single_diagnostic`とCPU `RuntimeBuffer`/streamを使う。従って「buffer APIを使わない」のではなく、**同じstandalone AQ4 matvec APIのCPU backendを使う**。layer 0 familyは各projectionを個別matvecしてhost recurrent helperへ渡す（`:1833-1994,2848-2881`）。multi-layer chainのlinear stateはcontext/layerごとにhostでcold resetし、self-attentionはhost causal attentionを直接計算する（`:2255-2400,2552-2666`）。

対してproduction M=1は`dispatch_token_for_phase`からresident layerの`run_device_step`へ入り、既定ではQKV/Z/A/B/gate/betaのfused API、outの`matvec_add`、fused MLPを使う（`qwen35_aq4_layer_runtime.rs:4950-4966,5090-5147`）。M>1は別のnative sequence pathである（`qwen35_aq4_model_runtime.rs:1020-1055`; `qwen35_aq4_layer_runtime.rs:4533-4816`）。

量子payloadの意味はCPU/HIPで揃っている。CPUはpacked nibble/codebook/group scaleを復元して要素ごとにscaleを掛けながら直列加算し（`runtime/src/ullm_runtime_parts/part_00.inc:2701-2735`; FFI CPU branchは`runtime/src/ullm_runtime_api_aq4.inc:477-503`）、HIPはgroup内のraw sumへgroup scaleを一度掛け、複数threadでtree reductionする（`runtime/src/ullm_runtime_hiprtc_sources.inc:623-729`）。これは代数的に同じでもf32の丸め順が異なる。さらにCPU fused APIはhost matvecを複数回呼ぶのに対し、HIPは専用fused kernelを使う（`runtime/src/ullm_runtime_api_aq4.inc:2221-2348`; `runtime/src/ullm_runtime_hiprtc_sources.inc:2093-2753`）。この差はH6の構成差ではなく、次段Phase 3cで検証すべきH5の実行差である。

### CPU-only追加検証

新規の大きな診断実装は追加しなかった。まずM=1/coldで既存artifactがすでに不一致を示すため、CPUだけでGPU数値差を模倣する実装を増やすより、既存unit testで制御契約を確認した。

| command | result | 確認できる範囲 |
| --- | --- | --- |
| `cargo test -p ullm-engine linear_attn_stateful_host_steps_match_full_recurrent --lib -- --test-threads=1` | `1 passed` | hostのtoken逐次conv/recurrent state更新とfull recurrent計算が一致する。full replayと同じtoken prefixのM=1継続をCPU数式上は支持する。 |
| `cargo test -p ullm-engine prefill_chunk_widths_cover_boundaries_and_tail_without_partial_progress --lib -- --test-threads=1` | `1 passed` | sessionがchunk幅境界/tailを完全なprogress単位で扱う。 |
| `cargo test -p ullm-engine native_session_commits_two_chunks_with_nonzero_cached_prefix --lib -- --test-threads=1` | `1 passed` | 2 chunk目をnonzero cached prefixとしてcommitするsession制御契約。 |

これらはすべてCPU-onlyで、GPU、resident service、systemd、active manifestには触れていない。C++ runtime build時には既存の`subobject-linkage` warningが出たが、各testは成功した。これらのtestはHIP kernelの数値等価性を保証しないため、その結論を過大に読まない。

### Phase 3b verdict

1. 07/14の`0.6151289249` path-oracleとattempt3 intermediate traceは、通常Gateway requestではなくdirect diagnosticである。この事実は、Gatewayとの差異を「07/14実測の説明」として使えないことを意味する。
2. direct artifact自身はcold/M=1で、case間resetを行い、step 0とreset後の別caseで既に`decoder_layer:0`不一致を観測した。従ってwarm session、request間cache reuse、M>1 chunk、RoPE/paged KVは、この既知の最初の不一致を説明できない。
3. normal GatewayのM>1 native sequence pathはharnessと異なり、通常online servingへ追加する未検証のH6リスクとして残る。しかしそれはM=1の既知差の根因ではない。
4. CPU standalone matvec/recurrentとGPU resident fused kernels/reduction orderの差が残る。よってこのスコープの結論は **「H6では07/14の既知不一致を説明できない。H5を有力に格上げする」** である。GPU kernel実行比較・fixは実施していない。

## 次の行動

- Phase 3bは完了とし、この監査だけを根拠にGPU kernelやproduction設定を変更しない。
- 次の明示的なPhase 3cで、まずM=1のCPU referenceとHIP M=1 fused pathをstage別に比較し、必要なら通常GatewayのM>1 native sequence pathを独立の追加寄与として扱う。
- 本作業ではGPU、active production service、systemd unit、active manifest、07/16の停止P3 harnessには一切触れていない。
