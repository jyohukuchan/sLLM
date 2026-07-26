# AQ4 P3 deployment delta analysis v0.1

Status: **分析完了。実装、build、evidence収集、manifest凍結、activationは未実施。**

調査時点の比較範囲は、live promotion source
`0cd760568e197e1adb4c4df3d6149591a912f709` から、HEAD
`6c7f4a63e647ae6e56b825ff4933e0ce07f834ba`（146 commits）までである。
後者へ最後に加わった `6c7f4a63` はSQ8 staging/LACTの文書だけを変更しており、AQ4
workerのbuild closureには入らない。本書は既存のファイル、git履歴、ソース、残存する
AQ4 build depfileを読むだけの調査である。

## Goal

P3のAQ4 prefill/decode性能成果をliveへ載せる際に、現行AQ4_0 workerからの正確な
deploy差分、SQ8_0 v2共有変更の影響、必要な再検証、最小リスクのsource cutを定める。

依頼BのAQ4-to-AQ4 runtime堅牢化（コード不変、path/ownershipのみ）とは意図的に
混ぜない。本書はB完了後に実施する性能candidate工程の入力であり、現行
`/etc/ullm/served-models/active.json`を変更する根拠でも許可でもない。

## Findings

### 基準liveと調査根拠

- 現行active manifestは `ullm.served_model.v2` / `AQ4_0` /
  `ullm.worker.v2`、worker SHA-256 `1f93f215...`、promotion source
  `0cd76056...`である。`journal/2026/07/24/aq4-bootstrap-closure-audit.md`
  は、active/candidateのbytes一致と当該workerのhash一致を記録している。
- AQ4 release directory
  `../uLLM-aq4-fidelity-promotion-release-f1a3cf4c/`にはworkerとlegacy engineの
  二つのbinaryだけが残り、名称が `build-receipt.json` / `build-provenance.json` の
  JSONは残っていない。代わりに、実際にこのworkerを作ったtarget directoryの
  `../uLLM-aq4-fidelity-promotion-build-target-f1a3cf4c/release/ullm-aq4-worker.d`
  とCargo fingerprintが、入力ファイルとfeature/rustflagsを保存している。本分析は
  このdepfileを一次のbuild-input記録として使った。
- sibling source worktree
  `../uLLM-aq4-fidelity-promotion-source-f1a3cf4c` は現在 `0cd76056...` にある。
  bootstrap auditが記すとおり、workerを作った時点のfidelity source `f1a3cf4c...` と
  promotion evidenceの `0cd76056...` の間にはRust/Cargo入力差分がない。そのため、
  `0cd76056...`をlive側の比較基準にしてもworker build closureを取り違えない。
- P3の連続範囲は first-parent上で
  `de0cd86e651ca1e7bad76acae0adac9216af0d47^..c4c9a9b344fc10e9a77ab0ded3293469d21b2f72`
  の47 commitsである。内訳はprefill 28 commits、decode 19 commitsである。
  この直前の14 commitsはP2運用/証跡であり、`ullm-aq4-worker`の実depfile closureを
  変更しない。
- P3の保存済みdirect API artifactには、2048 tokens / chunk width 128で
  `982.3834618154112 tok/s`を記録する
  `benchmarks/results/2026-07-19/qwen35-9b-aq4-production-opt-v0.1/p3/aq4-wmma-v4-promotion-and-e2e-v0.1/e2e.stdout`
  がある。ただしこれはsingle E2E windowであり、最終candidate identityにもP2の
  2 warmups + 10 measured matrixにもbindされていない。deploy proofとして流用不可である。
- P3 decodeのsummaryは `docs/plans/aq4-production-prefill-decode-optimization-plan-v0.1.md`
  とP2 runbookに `56.6%` と記録されている。一方、依頼にある
  `74.29 tok/s / 約131.2 tok/s / context 1371`の完全一致するraw resultは、調査した
  tracked text artifactからは確認できなかった。したがって本書ではその精密値を
  ユーザー提示の履歴値として扱い、新candidateの判定閾値には使わない。

### 実際の `ullm-aq4-worker` build closure

baseline depfileには、repository相対で80本のsource/watch inputがある。
`ullm-engine` source 56本、`ullm-runtime-sys`のRust/build input 5本、native input
19本である。HEADでは`lib.rs`が新しい`roctx.rs` moduleを取り込むため、source inputは
この1本を加える。`ullm-quant`はCargo metadataで`ullm-engine`のdependencyではないため、
そのRust/C++/build.rsはこのbinaryのclosureに含めない。

build metadataと設定は次のとおりである。

| 入力/設定 | baselineの観測 | base→HEAD |
|---|---|---|
| workspace `Cargo.toml`, `Cargo.lock` | workspace/package dependency解決 | 不変 |
| `.cargo/config.toml` | linker=`clang`、rustflags=`-C link-arg=-fuse-ld=mold` | 不変 |
| `crates/ullm-engine/Cargo.toml` | `ullm-aq4-worker` target、default feature | `18e0df01`で`autoexamples=false`とexample列挙を追加。AQ4 binaryのdependency/featureは不変 |
| `crates/ullm-runtime-sys/Cargo.toml` | runtime-sysのdefault feature | 不変 |
| `crates/ullm-runtime-sys/build.rs` | C++20/O2で`ullm_runtime.cpp`を`cc`でcompile。`ROCM_PATH`/compiler flagsをrerun inputとして監視 | 不変 |
| Cargo feature | `default`のみ。`rocm-ck-gfx1201`は未有効 | 不変 |

fingerprintにはdefault featureと上記mold rustflagsが記録されている。default buildでは
CKのHIP objectはなく、`ullm_runtime.o`のみが存在する。`sq8_ck_gfx1201.hip.cpp`などは
build.rsの`rerun-if-changed`対象だが、`rocm-ck-gfx1201` featureなしにはHIP compileされない。
P3のkernel本文は主に`ullm_runtime_hiprtc_sources.inc`に埋め込まれたHIPRTC sourceとして
実行時にcompileされる。

基準depfileの完全なrepository相対input inventoryは以下である。native欄の19本は
build.rsがwatchする入力であり、feature条件付きのCK sourceも含む。

```text
# crates/ullm-engine/src (56)
adapter_admission.rs                 adapter_fixtures.rs
aq.rs                                aq4_benchmark_worker_protocol.rs
aq4_benchmark_worker_runtime.rs      aq4_package_runtime.rs
aq4_worker_backend.rs                backend_dispatch.rs
backend_operation_registry.rs        bin/ullm-aq4-worker.rs
cpu_reference_executor.rs            decode_runner.rs
decoder.rs                           execution_batch.rs
execution_trace.rs                   format_id.rs
golden.rs                            host_bytes.rs
inference_api.rs                     lib.rs
loader.rs                            model_graph.rs
package.rs                            qwen35_aq4_head_runtime.rs
qwen35_aq4_layer_runtime.rs          qwen35_aq4_model_runtime.rs
qwen35_aq4_session.rs                qwen35_package_contract.rs
qwen3_loader.rs                      qwen3_names.rs
reasoning.rs                          scheduler.rs
served_model.rs                       session_worker_backend.rs
sq.rs                                 sq8_embedding_runtime.rs
sq8_generation_runtime.rs             sq8_layer_oracle.rs
sq8_layer_runtime.rs                  sq8_model_head_runtime.rs
sq8_sampling.rs                       sq8_serving_runtime.rs
sq8_stack_runtime.rs                  sq8_worker_backend.rs
sq8_worker_protocol.rs                sq8_worker_runtime.rs
sq_canonical.rs                       sq_optimized_reference.rs
sq_reference.rs                       sq_runtime.rs
state_schema.rs                       state_transaction.rs
verified_adapter_evidence.rs          worker_driver.rs
worker_protocol.rs                    worker_runtime.rs

# crates/ullm-runtime-sys (5)
build.rs                              src/lib.rs
src/lib_parts/part_00.rs              src/lib_parts/part_01.rs
src/lib_parts/sq8_ck.rs

# runtime native/watch inputs (19)
runtime/include/ullm_runtime.h
runtime/src/ullm_runtime.cpp
runtime/src/ullm_runtime_hiprtc_sources.inc
runtime/src/ullm_runtime_api.inc
runtime/src/ullm_runtime_api_core.inc
runtime/src/ullm_runtime_api_aq4.inc
runtime/src/ullm_runtime_api_linear_attn_prepare.inc
runtime/src/ullm_runtime_api_primitives.inc
runtime/src/ullm_runtime_api_sq8_0.inc
runtime/src/ullm_runtime_api_attention.inc
runtime/src/ullm_runtime_api_linear_attn.inc
runtime/src/ullm_runtime_api_smoke.inc
runtime/src/ullm_runtime_api_sq8_ck.inc
runtime/src/ullm_runtime_parts/part_00.inc
runtime/src/ullm_runtime_parts/part_01.inc
runtime/src/kernels/sq8_0/sq8_0_matvec_hiprtc.inc
runtime/src/kernels/sq8_0/sq8_0_matvec_runtime.inc
runtime/src/sq8_ck_gfx1201.h
runtime/src/sq8_ck_gfx1201.hip.cpp

# HEADで追加されたengine module input
crates/ullm-engine/src/roctx.rs
```

以下がこのclosureのうちbase→HEADで変わった全ファイルである。baseline inputに
交差する変更は24本、新規inputは`roctx.rs`、build metadata変更はengine manifest 1本である。

| 範囲 | 変更されたinput file | 代表commit | 影響の確認内容 |
|---|---|---|---|
| P3 | `crates/ullm-engine/src/aq4_package_runtime.rs` | `de0cd86`, `5acb228c`, `01a5da23`, `e6d81395`, `c4c9a9b3` | AQ4 GEMM/WMMA選択とdirect diagnostic wrapper |
| P3 | `crates/ullm-engine/src/aq4_worker_backend.rs` | `de0cd86`〜`e6d81395` | P3 HIP guard contractに6 capabilityを追加 |
| P3 | `crates/ullm-engine/src/backend_operation_registry.rs` | `de0cd86`〜`cb5e74c2` | runtime capability probe、implementation descriptor、production dispatchを追加/変更 |
| P3 | `crates/ullm-engine/src/qwen35_aq4_layer_runtime.rs` | `5fab4c6b` | paged causal GQA WMMA接続 |
| P3 | `crates/ullm-engine/src/qwen35_aq4_model_runtime.rs` | `b7b1e282` | P3 decode profiling marker接続 |
| P3 | `crates/ullm-engine/src/qwen35_aq4_session.rs` | `95ac8ebe`, `9460a9c5`, `b7b1e282` | prefill audit/GQA chunk/marker接続 |
| P3 | `crates/ullm-engine/src/session_worker_backend.rs` | `95ac8ebe`, `9460a9c5` | session dispatch/auditの整合 |
| P3 | `crates/ullm-engine/src/lib.rs`, `crates/ullm-engine/src/roctx.rs`（新規） | `b7b1e282` | ROCTx marker module。通常serveでは明示`enable()`なしにSDK DSOをloadしない |
| P3 | `crates/ullm-runtime-sys/src/lib_parts/part_00.rs`, `part_01.rs` | P3 WMMA/decode promotions | Rust FFI declaration追加 |
| P3 | `runtime/include/ullm_runtime.h` | `de0cd86`〜`c4c9a9b3` | WMMA/prototype C API、dispatch enumを追加（ABI versionは1のまま） |
| P3 | `runtime/src/ullm_runtime_api_aq4.inc` | `de0cd86`〜`c4c9a9b3` | AQ4 GEMM/WMMA/wide-load entry point |
| P3 | `runtime/src/ullm_runtime_api_attention.inc` | `5fab4c6b`, `9460a9c5` | paged causal GQA WMMA entry point |
| P3 | `runtime/src/ullm_runtime_api_linear_attn_prepare.inc` | `ac9b71af` | QKV-prepare shuffle prototype API |
| P3 | `runtime/src/ullm_runtime_api_primitives.inc` | `815b9a40`, `ac9b71af` | segmented RMSNorm/primitive API |
| P3 | `runtime/src/ullm_runtime_hiprtc_sources.inc` | P3 prefill/decode promotions、`c4c9a9b3` | HIPRTC kernel source本体 |
| P3 | `runtime/src/ullm_runtime_parts/part_00.inc`, `part_01.inc` | P3 prefill/decode promotions | host runtime wrapper/dispatch実装 |
| SQ8 v2共有 | `crates/ullm-engine/src/served_model.rs` | `90869be9` | v2 reasoning delimiterを各1 tokenへfail-closed化 |
| SQ8 v2共有 | `crates/ullm-engine/src/sq8_worker_protocol.rs` | `82d3658b`, `7c888c6d`, `90869be9` | shared v2 profile/released counter strictness。AQ4が`Sq8WorkerProfile`を直接使う |
| SQ8 v2共有 | `crates/ullm-engine/src/sq8_serving_runtime.rs`, `sq8_worker_backend.rs`, `sq8_worker_runtime.rs` | `82d3658b`, `90869be9` | shared v2 execution runtime。AQ4の`worker_runtime` alias経由の影響は下記の動的検証対象 |
| SQ8 v2共有 | `crates/ullm-engine/src/bin/ullm-aq4-worker.rs` | `90869be9` | 差分は`#[cfg(test)]` fixtureのsingle-token化のみ |
| SQ8 build metadata | `crates/ullm-engine/Cargo.toml` | `18e0df01` | `autoexamples=false`とexplicit examples。AQ4 targetのfeature/dependency/実行コードは不変 |

この表にないbaseline depfile inputはbase→HEADで不変である。特に
`reasoning.rs`、`worker_protocol.rs`、`worker_runtime.rs`、`Cargo.lock`、root
`Cargo.toml`、`.cargo/config.toml`、runtime-sys manifest/build.rsは不変だった。

## 分類表

| 分類 | commit群/代表commit | AQ4 workerへの事実上の影響 | 判定 |
|---|---|---|---|
| (a) P3 prefill最適化 | `de0cd86`〜`cb5e74c2`（28 commits）。代表: group16 WMMA `5acb228c`、paged GQA QK `5fab4c6b`、group8 `01a5da23`、linear recurrent `01717406`、direct output `ef62dc48`、ragged dispatch `67bd2a25`/`e6d81395` | header/API、HIPRTC kernel、registry、session、startup guard、AQ4 deployment profileを変更する | performance candidate本体 |
| (a) P3 decode最適化 | `b7b1e282`〜`c4c9a9b3`（19 commits）。代表: M1 wide-load `f746627c`、matvec-add wide load `a85305e9`、SiLU `76cfa761`、QKV `c747f3fb`、triple/RMSNorm/QKV-prepare `6df3680b`、segmented RMSNorm `6c55f7bd`、matvec-add `27b246df`、LM head `c4c9a9b3` | 同じruntime/HIPRTC sourceを更新。後半のpromotionは同じproduction symbolのkernel body差替えで、新guardを増やさない | performance candidate本体 |
| (b) SQ8 v2共有runtime | `82d3658b`, `7c888c6d`, `90869be9` | `served_model.rs`、`Sq8WorkerProfile`、shared worker runtimeを変える。AQ4 binaryはこれらをcompileし、一部を実行経路で使う | HEAD採用時のAQ4挙動/manifest受理リスク |
| (b) SQ8 v2 promotion/manifest | `f71bb2e5`, `8bf95a25`, `4de3aabb`, `127995b6`ほか | manifest freeze、format/protocol/schema selector、bundle routing、campaign admissionをfail-closed化する | AQ4 candidateの生成/activation手順に影響。ただしAQ4はbundle v1のまま |
| (c) P2運用、AQ5/importance、P3 test/diagnostic、文書 | P3直前14 commits、AQ5 `49fceeeb`〜`f24d0b06`（34 commits）、SQ8 track内の残りのtool/doc、`291ba215`、`6c7f4a63` | 実depfile closureにAQ5/importance由来の差分はない。P3の専用test parts/new binary/diagnosticはworker binaryのinputではない | worker buildには非影響。ただしpromotion tool/documentは必要時に別途選択する |

P3について「外部ABI/dispatch不変」と一括で表現することは正確ではない。
P3時点でAQ4 worker JSON wire protocol、`served_model.rs`、worker mainは変更されておらず、
外部worker protocolは不変である。一方、native ABI versionは`1`を維持しつつ、headerには
additiveなWMMA/prototype APIsとdispatch enumが加わり、registryとstartup guardは意図的に
新selectorへ変わる。したがってこれは後方互換なAPI追加だが、dispatch不変ではない。
「同じproduction symbolのkernel bodyを差し替え、guardも不変」という限定的な説明は、
decode後半のpromotionにのみ当てはまる。

## 影響評価: SQ8 v2とlive AQ4 manifest

### 静的schema/selector受理

HEADの`served_model.rs`はv2 manifestのtop-level、`format`、`worker`、`promotion`、
`reasoning`のkey集合をexactに検査する。AQ4 workerはさらに`AQ4_0`、
`qwen35_aq4_rdna4_v1`、`gfx1201`、`rdna4_aq4_resident`、greedy samplingを検査する。

現行live manifestのbytesは、SQ8 v2由来の静的strictnessには適合する。

- `format.format_id=AQ4_0` と正しいimplementation selectorを持つ。
- `worker.protocol=ullm.worker.v2`であり、v2 reasoningを持つ。
- `start_token_ids=[248068]`、`end_token_ids=[248069]`、
  `forced_end_token_ids=[248069]`であり、`90869be9`が導入した「各delimiterは厳密に
  1 token」条件を満たす。
- `reasoning`のbudget/reservation、greedy sampling、worker identityもAQ4 contractと整合する。

従って、SQ8 v2のformat selector必須化、one-token delimiter、schema exactnessだけを
理由にlive AQ4 manifestを拒否することはない。`reasoning.rs`自体はbase→HEADで不変である。

またHEADの`tools/activate-served-model.py`はformatでfail-closedにrouteし、AQ4_0には
`ullm.generic_reasoning_release_bundle.v1`、SQ8_0にはbundle v2を対応付ける。P3 AQ4は
SQ8 v2 bundleへ移行しない。profile生成時にはAQ4_0 / `ullm.worker.v2` /
`ullm.aq4_resident_promotion.v1`の組合せを使うが、
`promotion.required_schema_version`はprofile専用である。最終manifestの`promotion`は
`source_commit`、`receipt`、`receipt_sha256`の三key固定なので、selectorを最終JSONへ
追加してはならない。

### HEAD/P3 workerを起動する場合の受理

上の静的判定は、現行manifestを新workerへそのまま渡せることを意味しない。これは**不可**である。

1. active manifestは旧releaseの絶対`worker.binary`とSHA-256 `1f93f215...`をbindingする。
   loaderはfile hashを検査し、worker起動時には`current_exe == worker.binary`も要求する。
   P3/HEADでbuildしたworkerはpath/hashが異なるため、現行bytesを再利用できない。
2. 現行manifestの`required_environment`は30件である。P3 workerの
   `validate_resident_model_contract`はlistを順序非依存で正確に一致させ、P3後の正しい
   contractは36件である。現行listにない次の6件があるため、P3/HEAD AQ4 workerは
   guard mismatchでfail-closedになる。

```text
ULLM_REQUIRE_HIP_AQ4_REGISTER_BM8_GROUP8_KERNEL  # de0cd86
ULLM_REQUIRE_HIP_AQ4_WMMA_GEMM_KERNEL            # 5acb228c
ULLM_REQUIRE_HIP_AQ4_WMMA_GEMM_GROUP8_KERNEL     # 01a5da23
ULLM_REQUIRE_HIP_AQ4_WMMA_GEMM_RAGGED_M_KERNEL   # 075c7f6e
ULLM_REQUIRE_HIP_AQ4_WMMA_GEMM_GROUP8_RAGGED_M_KERNEL # e6d81395
ULLM_REQUIRE_HIP_PAGED_CAUSAL_GQA_WMMA_KERNEL    # 5fab4c6b
```

将来のP3 manifest/profileには、少なくとも新release workerのabsolute path/SHA-256、
正確な36 guards（上記6件を含む）、新candidateの`promotion.source_commit`、receiptと
そのhashをbindする必要がある。既存product/tokenizer/generation fieldsを省略・拡張して
よいという意味ではない。manifest generator/loaderの物理file hash検証、gateway経由で各
guardが`=1`になること、実wireの動作は本調査では実行していないため未確認である。

`82d3658b`/`7c888c6d`/`90869be9`のshared v2 runtimeはAQ4から完全にdeadではない。
AQ4 binaryは`Sq8WorkerProfile`を直接importし、shared `worker_runtime` aliasを通る。
released eventの`reasoning_tokens`/`forced_end_tokens`必須化もある。現行v2 reasoningとの
静的整合は確認できるが、direct worker/SSE/Gatewayの実行検証なしにwire互換を断言しない。

## 検証要件

以下は今後P3 candidateをliveへ載せる場合の要件定義であり、本タスクでは実行していない。
既存P2のold evidenceはfinal candidate identityにbindされず、P3 direct E2Eはsingle window
なので、流用しない。

### Fidelity: 必須8指標と例外の扱い

`docs/plans/aq4-fidelity-root-cause-and-fix-plan-v0.1.md`と
`docs/proposals/aq4-p2-fidelity-holdout-protocol-v0.1.md`に従い、calibration 24 +
formal holdout 24の48 rowsで次を再測定する。

| 指標 | 必要条件 |
|---|---|
| `logits_relative_l2` | ≤ 0.1468 |
| `logits_cosine` | ≥ 0.9856 |
| `hidden_relative_l2` | ≤ 0.1916 |
| `hidden_cosine` | ≥ 0.9800 |
| `topk_overlap_rate_k10` | ≥ 0.7900 |
| `bf16_top1_retained_in_aq4_top10_rate` | Wilson下限 ≥ 89.9% |
| `hidden_max_abs` | 必須記録、診断専用で閾値なし |
| `token_agreement_rate` | Wilson下限 ≥ 89.9% |

各rowのrelative-L2が1.0を超える、non-finite、identity/split不一致は構造的No-Goである。
過去の`token_agreement_rate`は20/24、Wilson下限0.676で唯一閾値未達だったが、margin分析後に
ユーザーがAQ4量子化ノイズとして許容済みである。新candidateでも必ず値を再測定・記録し、
この承認済みの非blocking例外として明示する。黙ってPassとはしない。

この例外はcandidate-vs-activeのexact behavioral gateを緩めない。token、ordered top-k、
KV/cache、position、scheduler、reset/cancel等の完全一致は別の即時No-Goである。さらに
凍結policyとhistorical binding specの矛盾は`formal_p2_status=blocked_contract_resolution`
として残るため、正式な文書上の解決も記録する必要がある。

### 性能・状態・統合gate

- P3 correctness: shape/dtype/finite、hidden/logit、greedy token、top-k、
  KV/recurrent/conv/cache/position/chunk boundary、cancel、publish failure、EOS/length、
  resetと次request、OOM、unexpected fallback、workspaceを確認する。
- normal performanceは各resident caseにつき2 warmups + 10 measuredで、p50/p95、TTFT、
  ITL、VRAM peak、fallback countを保存する。prefill p50が5%超、p95が10%超の回帰はNo-Goである。
- P4: prefill n1011 ≥ 318.19 tok/s、n2048はold tokenwise baseline比5倍以上かつOOMなし、
  n1024の1000 tok/sは報告値であり必須閾値ではない。context 1339 decodeとshort decodeの
  5%超回帰を許容しない。
- P5: c1339 ≥ 53.3 tok/sかつ同一identity active baseline非回帰、c16/c128/c512のp50が
  5%超回帰しないことを確認する。profileだけでなくnormal throughputでも改善を確認する。
- P6: offline full-model、direct worker non-stream/SSE、Gateway API/SSE、全reasoning mode、
  OpenWebUI Stop/worker failure recovery、100-chat soak、restart後20-chat、bundle/rollback、
  canary/post-activation probeの順で同一candidate identityを確認する。

### R9700窓の回数と条件

P2 performance runbookが規定する最小構成は、serial single-use R9700 window 28本である。

| 種別 | 本数 |
|---|---:|
| normal prefill（prompt 7長） | 7 |
| normal decode（context 7長） | 7 |
| detailed rocprof | 6 |
| full-vector target-path oracle | 8 |
| **性能小計** | **28** |
| 48-row fidelity service-stop window | **1** |
| **文書上の最小測定/service窓** | **29** |

fidelity windowの前には3回のvalid R9700 guard rehearsalも必要である。P6 integration/
release/rollback/canaryの窓数は既存文書に固定値がなく、29より増えるが正確な合計は未確認である。
既存実績から確定している時間は48-row fidelityの約13.5分だけであり、28性能窓の各時間・
総時間はrunbookに規定がないため推測しない。

すべてのR9700 GPU窓（guard rehearsalを含む）では、`ullm-openai.service`を止める/lockを
取得する**前**に次を行い、`inactive`をartifactへ記録する。boot-disabledだけでは不十分であり、
teardownでllama serviceを再起動しない。

```text
systemctl stop llama-qwen35-udq4.service
systemctl is-active llama-qwen35-udq4.service  # expected: inactive
```

過去P3の`982 tok/s`/decode `56.6%`はllama comparison baselineが常駐した条件で得られ、
junction 85°C、core clock最大固定、VRAM 5.3 GB占有だった。以後の値はstop後の温度・clockを
記録し、絶対値比較にこの熱条件差を必ず注記する。

既存P2 baselineも完全にはsealされていない。記録済みのnormalは13/14、detailed profileは6/6、
path-oracle 8本は未着手で、baseline JSONL sealはdeferredである。active identityと異なる場合は
runbook自身が`separated_not_comparable`とするため、final P3 candidateでidentity-boundに取り直す。

## Risks

1. P3は性能カーネルだけでなくnative API追加、registry dispatch、capability probe、fail-closed
   guardを変える。kernel differentialだけでなくstate、fallback、startup guard、manifestを検証しないと
   速度向上を安全性の根拠にできない。
2. HEADを採用するとP3後のSQ8 v2 shared runtime/manifest strictnessも同時に採用する。AQ4 direct
   workerのprofile/released event/SSE回帰時に、性能差分とshared runtime差分を分離できない。
3. 現行manifestはschema上はv2 strictnessを通る一方、old binary identityと30 guardsに固定されている。
   新binaryと新manifestを別々に扱うと必ずfail-closedまたはidentity mismatchになる。
4. 既存P3のdirect 982 tok/sとdecode summaryは、熱条件、matrix不足、candidate hash未bindのため、
   production performance guaranteeではない。
5. Bのpath/ownership堅牢化をP3 source/manifest差分に混ぜると、回帰が起きた際に原因を
   堅牢化か性能変更かへ分解できない。

## Recommendation

**P3成果だけの技術的な切り出しは可能であり、HEAD全体の採用ではなく、連続するP3終端
`c4c9a9b344fc10e9a77ab0ded3293469d21b2f72`を候補sourceとして採用することを推奨する。**

根拠は次のとおりである。

- `0cd76056...`からP3終端までの47 commitsは連続し、P3後HEADはP3計算input 18本と
  `roctx.rs`を一切変更していない。
- P3前の14 commitsにもAQ4 worker depfile closureの差分はない。
- P3終端はprefillとdecodeの両方を含み、AQ5/importance 34 commitsとSQ8 v2 49 commitsを
  含まない。
- HEADにはAQ4から到達しうるshared v2 runtime/manifest変更が含まれるため、性能昇格としては
  余分な回帰面を増やす。

方法はpromotion commitだけを恣意的にcherry-pickすることではない。前段prototype、direct API、
validation、registry integrationに依存があるため、B完了後に`c4c9a9b...`のclean detached
worktree（またはこの連続rangeだけの専用branch）からcandidateを作る。さらに小さく19本のruntime
sourceだけをsquash再構成する方法は、新しい未検証patchを作るため推奨しない。

このP3-only sourceでも、新しいrelease worker、36 guardsを含む新profile/manifest、receipt、
全identity binding、上記のfidelity/performance/integration再検証は必要である。AQ4はgeneric
bundle v1 routeを維持し、SQ8 bundle v2を導入しない。

## Next Actions

1. 依頼BのAQ4-to-AQ4 runtime堅牢化を完了し、P3性能candidateとの差分を別commit/別evidence
   として固定する。
2. B完了後、`c4c9a9b...`をclean sourceとして新release candidateを作る。旧
   `qwen35-9b-aq4-reasoning-f1a3cf4c.profile.json`はold binary pathを含むため再利用せず、
   P3の36-guard contractを正確に持つ新profile/manifestを作る。
3. 新candidate identityで、29以上の既定R9700測定/service窓、fidelity例外の明示記録、
   direct worker/Gateway/SSE/OpenWebUI/rollback検証を行う。各窓でllama comparison baselineの
   `inactive`、温度、clockを記録する。
4. release bundle/rollback bindingを独立検証し、軽量昇格方針の生成品質確認と
   ロールバック準備が完了するまで`active.json`の実バイトを差し替えない。

## 2026-07-26 P3 deployment execution record

この記録では、共有 worktree の動く `HEAD` を候補にせず、上記推奨どおり P3 終端
`c4c9a9b344fc10e9a77ab0ded3293469d21b2f72` を選んだ。`0cd76056..0455b119` の
255-commit snapshot を再確認し、47 本の連続 P3 commit（prefill 28、decode 19）が
この endpoint に入ることを確認した。AQ5/importance と後続の SQ8 実験は含めていない。
新しい detached worktree から `ullm-aq4-worker`、prefill timing、decode profile のみを
`CARGO_BUILD_JOBS=16` で build し、共有 HEAD は移動していない。

候補 worker SHA-256 は
`ba8c46d6eee81d508f4b2e744ec05d8743a46bf44100ec66257c8d8ae739e265`、候補 manifest
SHA-256 は `a98910dc5bf59dc768e5bcd20bcf58968699540eb1b33df33066dcb6f274fe49` である。
新しい release は既存 release を上書きせず
`/opt/ullm/aq4-p3-deployment-v0.1/releases/aq4-p3-c4c9a9b3/` に作成した。候補は active
production と同一の protected product package manifest
`a790a033f57d9c5b9ae0d731a463c26b86aec691f771ce88bb543d676f08e5ad` を使用し、36 guard
contract（旧 active の 30 に P3 の 6 guard を追加）を validation 済みである。

BF の config-driven loader 系も独立に再確認した。Qwen3.5-9B `config.json`
（SHA-256 `d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05`）は
`Qwen3_5ForConditionalGeneration` / `qwen3_5_text`、32 layer、
`linear_attention, linear_attention, linear_attention, full_attention` の 8 回反復である。
`b21b2723` は AQ4 load path に届くが、P3-only release には意図的に含めず、BF の実行成功は
互換性の裏付けとして扱った。SQ8 側が完全に無関係とはいえず、`82d3658`、`7c888c6`、
`90869be` の v2 shared worker/reasoning runtime は AQ4 worker にも到達し得る。しかしこれらは
選択 source より後なので candidate には入らない。残る SQ8_0 tile/probe/gate 実験も同様に除外した。

R9700 (`gfx1201`) だけを使う isolated direct timing は、prefill 2,048 token / chunk 128 で
**970.6107 tok/s**（既報 982.3835 より -1.198%）、decode C=1339 / 32 measured step で
**73.4568 tok/s**（既報 74.29 より -1.122%）だった。したがって 982.4 tok/s の厳密再現では
ないが、同一 product/package/profile で近傍の P3 水準は再現した。歴史測定は junction 約85°C、
最大固定 clock、5.3 GB の llama comparison 常駐という別条件であり、絶対値同一の主張はしない。
historic decode 56.6% の raw theoretical denominator は保存されていないため未確認であり、
昇格 gate に用いない。

軽量昇格の初回は generic `tools/promote-served-model.py` の preflight と baseline readiness を
通過したが、baseline third request 中の 21:45:59.160 JST に別 session の
`systemctl stop ullm-openai.service` が割り込んだ。worker EOF/HTTP 500 と残る transport failure は
この teardown に対応する。tool は `baseline_failed_before_mutation` で fail-closed となり、
candidate bytes は一度も active にならなかった。初回 evidence は
`benchmarks/results/2026-07-26/aq4-p3-deployment/lightweight-promotion-attempt-1/` に保存した。

この中断後、BH が 21:46:18 JST から `/run/ullm/r9700.lock` を保持する decode-attention window を
開始した。service をその lock と競合して起動すると `WorkerBusy` になり StartLimit を浪費する。実際に
別 session の 21:57:48 JST start と systemd retry 2 回がこの lock に衝突し、21:58:28 JST に
`StartLimitBurst=3` が尽きた。したがって lock 解放と 15 分 StartLimit window（保守的に
22:13:29 JST 以降）の経過、active manifest SHA が期待値
`c57a2b6c5827b8ddd102560b3f5efd879711705cf4d8a36f4d7872821d05fca4` のままであることを確認するまで
generic route を再実行しない。最終昇格または no-go の結果はこの条件後の actual text comparison に
基づき追記する。

### Final promotion outcome

BH が lock を解放し、StartLimit の時間窓を越えた後、active SHA が依然
`c57a2b6c5827b8ddd102560b3f5efd879711705cf4d8a36f4d7872821d05fca4` であることを再確認して
fresh evidence directory で generic route を再実行した。old active から固定 10 prompt の実生成を
すべて保存し、atomic swap、成功した service restart 1 回、bounded readiness retry、candidate から
同じ 10 prompt の実生成をすべて完了した。

比較は `blocking_findings: []`、10/10 nonempty response で PASS だった。日本語/英語説明、Python/
JavaScript code、要約、multi-turn、翻訳、reasoning のいずれにも空応答、文字化け、反復、code 要求の
放棄、途中 abandon は検出されなかった。今回の deterministic suite では診断上 exact output match
1.000 になったが、これは top-1/logits gate として用いず、保存された actual text の品質で判定した。
candidate readiness は health/ready/models がすべて200になるまで bounded retry を10回行い、最終 probe
で成功した。service event は `restart` 1 回、`start_limit_recovery: false` である。

したがって `AQ4_0` P3 candidate は **activated** である。新 active manifest SHA-256 は
`a98910dc5bf59dc768e5bcd20bcf58968699540eb1b33df33066dcb6f274fe49`、worker SHA-256 は
`ba8c46d6eee81d508f4b2e744ec05d8743a46bf44100ec66257c8d8ae739e265` である。post-activation の
`ullm-openai.service` は active/running / `Result=success` / `NRestarts=0`。rollback tool を
`--yes` なしで実行し、旧 manifest `c57a2b6…fca4` への strict-byte rollback preflight `ready: true` を
確認した。rollback は不要なので実行していない。生成文、comparison、transaction、rollback preflight は
`benchmarks/results/2026-07-26/aq4-p3-deployment/lightweight-promotion-attempt-2/` に保存した。

なお、activation 完了後に BJ の SQ8_0 isolated measurement が service を一時停止したが、candidate
manifest は不変で、BJ の restore 後に `/readyz` HTTP 200、running worker の executable hash
`ba8c46d6…e265`、active/running / `Result=success` / `NRestarts=0` を再確認した。この後続 stop は
P3 promotion の failure や rollback ではない。

さらに 22:35:50 JST に、前記 restore 後の別 BJ `SQ8_0` `--speed-first` window が開始し、22:35:52
JST に gateway を停止して R9700 lock を保持した。22:41 JST の handoff 観測では manifest は依然
`a98910dc5bf59dc768e5bcd20bcf58968699540eb1b33df33066dcb6f274fe49` だが、gateway はその他 window
の所有下で意図的に `inactive/dead` / `Result=success` / `NRestarts=0` だった。AQ4 deployment は lock と
競合する start を行わず、この window の trap が restore を担当する。これは既に完了した P3 activation、
text quality 判定、rollback preflight の結果を変更しない。
