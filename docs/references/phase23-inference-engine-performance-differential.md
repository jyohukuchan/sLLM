# Phase 23 inference-engine performance differential

> 固定日: 2026-08-18
> 用途: performance探索の技術事実。非llama engineの実装sourceとして使用しない。

## 比較境界

Phase 23の数値peerは、Phase 5で固定したllama.cpp commit
`f5919bf458ef190468b5c329bb293f8a54a1e69c`と同一source revisionのQwen3.5-4B BF16 artifactである。
256-token prefillはE1 system-equivalent、fresh decodeはtoken列と出力長が異なるためE2 diagnostic-onlyとした。
vLLMとSGLangは固定sourceの構造比較に限定し、速度比には使用していない。

vLLM revision `568afb3a13806beb53bb2e6bd518269357b237c0`とSGLang revision
`fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1`はlocal reference treeへ固定されている。ただしPhase 23環境には
両Python moduleも対応container imageもなく、新たなPyTorch/ROCm stackを構築すると事前固定controlではなくなるため、
実行比較をE2 technical comparisonへ限定した。再現確認は次の3 commandで行った。

```text
python3 -c 'import vllm'
python3 -c 'import sglang'
podman images --format '{{.Repository}}:{{.Tag}}'
```

## sLLMで確認したcritical-path差

### Prefill terminal projection

sLLMのgeneration adapterはprefill結果の`token_ids().last()`だけを次tokenとして使う。一方、Qwen graph loweringは
`M`行のhidden state全体をvocabulary projectionへ渡し、terminal argmaxも`M`行分実行する。256-token profileでは
`[256, 248320]`相当のLM-head-shaped dispatchがdevice timeのV620 13.48%、R9700 46.92%を占めた。
これはpeer実装のcopyではなく、sLLMのsourceとtraceから独立に見つけた余分な処理である。

実装候補は、通常generation pathだけで最終hidden rowへsliceしてから一行のLM head/argmaxを実行することとする。
all-logits、MTP hidden/logits、training用途に相当する複数行契約は明示pathとして保持し、first-token/logits oracleと
255/256/257境界で反証する。

### Decode matrix family

fresh long decodeはV620 32.43 tok/s、R9700 36.99 tok/sだった。Qwen profile履歴とfresh Gemma 4 controlの双方で
matrix workが支配的だが、Phase 22で単一M=1 shapeを速くしてもfull-model wallへ転化しなかった。したがって次候補は
単一kernel tuningではなく、gate/upやQKVなどprojection family単位のshared load/fusion、launch plan replayを測る。
Gemma 4はmixed low-bit providerであり、BF16 kernelをそのまま共通化する根拠にはしない。

### Service scheduling

sLLM serverのruntimeは一つのFIFO channelと一つのworker loopを所有し、一要求のgeneration完了までawaitする。
Q1の同時2要求は両GPUでほぼ完全に直列化し、HTTP/SSE residualは約0.5〜0.6 msだけだった。

固定vLLM sourceのschedulerはwaiting/running集合とtoken budgetを管理する。固定SGLang sourceのschedulerはrunning batchへ
new workをmergeし、overlap loopを持つ。これらはcontinuous batchingが既存engineで成立するという技術事実の確認だけに使い、
source expression、control flow、testはsLLMへcopy/adapt/portしない。sLLMでの候補は既存request/KV/cancellation contractから
独立に設計し、Q1/Q2 aggregate throughputとper-request順序で評価する。

### Cold model load

sLLMのfresh-process model-ready中央値はV620 10.53 s、R9700 11.60 sだった。固定llama.cpp controlのmodel loadは
1.18 sである。sLLM sourceでは起動ごとのfull GGUF SHA-256、bindingごとのresident allocation、chunk uploadごとの即時completion
waitを確認した。trace上のH2D合計はV620 0.54 s、R9700 0.71 sであり、残差の多くはhost verification/read/allocationと
直列upload orchestrationにある。identity cacheはcontent mutation検出を弱めないことを採用条件とする。

## 採用しない差分

- HTTP JSON encode、non-stream response、SSE framingは現状約0.6 ms以下であり、主要最適化候補にしない。
- profiler wallはobserver effectを含むためproduction E2E比較に使わず、device shareとdispatch mechanismだけに使う。
- fresh sLLM decodeと固定llama.cpp decodeはtoken列・出力長が異なるため、勝敗ratioを作らない。
- vLLM/SGLangは実行環境が一致せず、architecture上の差だけを記録する。

## Source locations and provenance boundary

- sLLM generation consumption: `crates/sllm-frontend/src/generation.rs`, `GenerationExecutorV1` prefill path。
- sLLM graph/provisioning: `crates/sllm-core/src/qwen_execution.rs`, graph lowering、semantic execution、resident upload path。
- sLLM GGUF verification: `crates/sllm-core/src/gguf_writer.rs`, derived GGUF verification path。
- sLLM service serialization: `crates/sllm-server/src/runtime.rs`, FIFO worker loop。
- vLLM technical fact: `reference/vLLM/vllm/v1/core/sched/scheduler.py` at the fixed revision above。
- SGLang technical fact: `reference/SGLang/python/sglang/srt/managers/scheduler.py` at the fixed revision above。

本調査による第三者code importはない。llama.cppは既存のimmutable performance evidenceだけを使用し、vLLM/SGLangは
facts-only inspectionとした。実装候補はPhase 24以降で別途採否し、直接reuseが生じる場合は通常のprovenance手順を適用する。

[Phase 23集計](../../ci/matrix/phase23-performance-discovery-summary-v1.json)
[Phase 23 history](../history/2026/08/11-20/phase23-cross-engine-differential-performance-discovery.md)
