# Phase 46履歴: conversion・quantization・benchmark・品質評価tool

## 2026-08-27: 実装

- `sllm-tools` workspace crateと`sllm-artifact`、`sllm-bench`、`sllm-eval`を追加した。共通
  `sllm-phase46-tool-run-v1` manifestはsource/output/raw evidence、recipe、tool commit、実行binary SHA-256、
  args、OS/architecture、compile-time Rust toolchain、model/dataset identityを結合し、0件選択、空file、
  non-finite値、未知/all-zero commit、未知必須fieldをfail closedにする。
- `AtomicBundleV1`はfresh hidden staging、file fsync、Linux `renameat2(RENAME_NOREPLACE)`、parent fsyncでbundleを
  一括publishする。single JSONはhard-link no-replaceでpublishし、parent fsync失敗時は完成名をrollbackする。
  debug dumpはdrop/error時にpartialを除去する。
- reviewed Qwen3.5 capability経路へGGUF tensor-boundary split/exact merge、Phase 45互換LoRA conversion、MXFP4/NVFP4
  repack、FP8/NVFP4/MXFP4 quantize、deterministic F64 sum-of-squares imatrixを実装した。unknown architecture/dtype、
  一般Q8_0/Q4_K、malformed/truncated/duplicate/foreign part、non-finite inputはunsupportedまたはerrorのままとした。
- `sllm-bench aggregate`はwall/GPU timing、warmup/measured/rejected、E2E/TTFT/TPOT/prefill/decode、resourceの
  measured/unsupported/missing、fallback/cleanupを別fieldで保存する。`sllm-eval`はperplexity、KLD/top-1/logit差、
  task、long-context coverageをbounded inputで評価する。debug dumpは既定無効、16 MiB等のhard limitとclosed metadata
  allowlistを持ち、prompt/response、credential、pointer、model payloadを受理しない。
- 既存`sllm-convert-gguf`へ`--output-bundle`を追加し、GGUF、`derived-gguf-lock-v1`、共通run manifestを一つの
  transactionでpublishするようにした。legacy二file経路は複数出力をatomicにpublishできないため拒否する。

## 2026-08-27: Qwen3.5変換証拠

- commit基点`dbc7c379943c6176abc22decf617f554f8e1c356`のdirty integration treeから、reviewed
  `Qwen/Qwen3.5-4B` BF16 lock fingerprint
  `sha256:f143d7b504170d071c77818105f7a07dc0297c6bea0c61a5404b071fed0c1fae`とverified snapshotを使い、
  738 tensorのGGUFをrepository外へ生成した。これはrelease identityではなくPhase 46 integration evidenceである。
- bundleは`/home/homelab1/.cache/sllm/derived/phase46-qwen35-bf16-bundle`にatomic publishされ、GGUFは
  `9,343,583,936` bytes、SHA-256 `c571c54eb8e2c9e935790d885e6d20f29c5fc82cd00ae28ddb5937a77c7fc675`、
  metadata SHA-256 `7a5149817a3ffb644ab9a8dd3ddd503faee431c4d1364a3c25b8a1024cbf05b4`、tensor catalog SHA-256
  `4f909053cc8318cbe18a809a9107efc4330ba1fcdbc7d82556ad67caf2711d44`だった。
- derived lock fingerprintは
  `sha256:d553db4d10df5655b681b067ac0e8359defe85ab384e805c97f8a296854b4c12`、file SHA-256は
  `821e43dc1c568f4c5b0fdea8d831a15177a6c652e9f5c0390b5aba0b99b47547`、run manifest SHA-256は
  `af71f86aa63e58f27e178e1ffee81967e44e5c86692bfa916e170908e0c4e5f7`である。model、GGUF、raw traceは追跡しない。

## 2026-08-27: KV品質policyと検証

- project-authored CC0 token-ID fixtureをseed 1729、順序固定、長さ
  `1/15/16/17/255/256/257/511/512/513`で固定した。dataset SHA-256は
  `a2252d882ffd7e1fbb546d86b2b573bd2410467382c7da874f4fbd3dc8adc77d`で、K/V、early/middle/tail、
  layer、KV head、block tailのcoverage metadataを持つ。
- `kv-cache-default-v1`はcandidate観測を含めず、Qwen3.5-4B BF16、dataset、metric、3反復、threshold、
  inclusive/exclusive境界、missing/non-finite/0件FAIL、target独立判定をfreezeした。Phase 53候補binary、derived lock、
  candidate reportは未作成なのでnull／`insufficient-evidence`のままであり、新KV形式やdefault採用をPhase 46のPASSへ
  読み替えない。policy自身は循環を避けてself digestを持たず、consumerがcanonical file bytesを外部SHA-256で結合する。
- ROCm 7.14.0、HIP `7.14.60850`、LLVM 23、Code Object V6、wave32のexact `gfx1030` release buildで、10 caseを
  baseline＋3 measuredとして実行した。全requestがHIP-only、fallbackなしで、baseline対firstのKLD/max logit deltaは0、
  top-1 agreementは1.0、3反復のNLLはcaseごとに一致した。aggregate FP16 perplexityは10 selected next-tokenで
  `235993.6527695604`、long-context coverageは1.0、model residentは`8,411,592,192` bytes、shutdown後のresident/request/
  workspace、retryable cleanup、durable quarantineは全て0だった。
- 統合review後のrunnerは各caseでprefill最終rowに加え1-token decode継続を実行し、各境界でcommitted KVを読み出す。
  exact `gfx1030` binary SHA-256は`25b6ddc8227c9ed3f55075a6457b3e3409c1bd70d3cd9d15521c2776793178b8`、
  repository外v2 report SHA-256は`6276fbab00d50bff0618467537eec0d02f4d9061a950e9e6ab75d12dfb14f934`でschema PASSした。
  prefill/decodeのcomparisonは20/20 top-1一致、KLD/max logit delta 0で、旧prefill-only reportはpolicyから外した。
  synthetic expected-next task scoreは0であり、単独のsemantic task品質証拠にはしない。Phase 53は全metricの独立判定を必要とする。
- 同じsourceからのexact `gfx1201` buildはR9700でembedding launch時に`device kernel image is invalid`となったためGPU PASSへ
  昇格せず、gfx1201 baselineとcandidate decisionを`required-before-candidate`／`insufficient-evidence`のまま維持した。
  gfx942もPhase 46の同一baseline runnerでは未測定であり、別targetのgfx1030 PASSで代用しない。
- baseline evidenceを結合したpolicy file SHA-256は
  `3e8b1696ebfd485606762d9b3c07fd2694f6157abf43745eabbbc2913240cb1d`である。Phase 53 reportはこのexact digestを参照し、
  policy変更後のcandidateへ旧digestを流用しない。

## 2026-08-27: host gatesと既知制約

- `cargo test -p sllm-tools --locked --offline`はunit 6、artifact 5、CLI 4、quality/debug 10をPASSした。
  CLI testは全binary help/capability、unknown/unsupported/zero選択、1x33 tail quantize bundleを実processで検証した。
- `cargo clippy -p sllm-tools --all-targets --locked --offline -- -D warnings`、baseline binaryのclippy、format、
  Phase 46 schema/policy test 7件、full JSON manifest/matrix validator、Markdown link validatorをPASSした。
- 一回の累積integration reviewはsource binding、schema/producer parity、sample order、gfx942 target、atomic publish、
  long-context decode継続の指摘を修正後にfocused再確認し、correctness/security blocker 0でPASSした。
- 一般architecture、Q8_0/Q4_K、remote service、leaderboard、raw prompt、無制限logit、runtime providerのないrecipeは未対応である。
  Phase 46 toolのquantize/repack成功はproduction runtime supportを宣言しない。KV block16形式、kernel、selector、target別default
  採否は[Phase 53保存済み計画](../../../../plans/archive/2026/08/21-31/phase53-kv-fp8-block16-default-adoption.md)が所有する。

[対応する計画](../../../../plans/archive/2026/08/21-31/phase46-conversion-quantization-benchmark-quality-tools.md)
