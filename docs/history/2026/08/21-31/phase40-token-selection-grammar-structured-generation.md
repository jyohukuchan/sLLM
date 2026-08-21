# Phase 40 token selection・grammar・structured generation

## 目的と到達点

Phase 40は、profile-v1の既存greedy互換を維持したまま、sampler chain、bounded grammar、structured output、logprob metadata、
複数choice stateをgeneration contractへ統合する作業である。2026-08-21時点でA0〜A3、B1〜B3、C1〜C2、D1〜D2のhost/API/HIP/
model-adapter実装とPhase Eを完了した。V620/R9700のselector contract matrix、gfx942 compile/route、Qwen/Gemma sampled structured
generation、最終workspace・integration reviewをPASSした。本historyはcloseout時点を記録する。

llama.cppは意味論と比較対象として参照したが、Phase 40でそのsourceを直接copy/adapt/portしていない。したがって新規provenance
blob、license notice、pending import commitはなく、既存の参照source lockとprofile-v1 oracleだけを維持する。

## 実装内容

- `SamplerChainConfigV1` と `SamplerChainV1` を追加し、legacy greedy/temperature/top-p/presence-frequency semanticsを保ったまま
  top-k、min-p、typical、repeat、dynamic temperature、ignore-EOS、DRY、XTC、Mirostat v1/v2、logprobsをversioned orderへ固定した。
  NaN/Inf、zero mass、all-masked、checked-bound failureはsilent fallbackせずerrorにする。
- tokenizerのraw bytes seam、bounded token trie、partial UTF-8 grammar state、GBNF parser/runtimeを追加した。generic JSON objectはdepth 1、
  containerあたり最大4 members/items（repeat 0..3）、string/number 64、whitespace 16にboundedし、JSON Schemaは別のglobal limitである。
  JSON objectと、
  object/array/string/number/integer/boolean/null、enum/const、required、`additionalProperties:false`、`anyOf`、local `$defs`/`$ref`
  に限定したJSON Schema lowererを実装し、unsupported keyword、remote/recursive reference、上限超過をrequest受付時に拒否する。
- OpenAI Chat Completionsへ`n=1..=8`、`logit_bias`、`logprobs`/`top_logprobs=0..=20`、`response_format`を接続した。choiceごとの
  seed/RNG、sampler/grammar/stop state、usage、non-stream/SSE indexを分離し、grammar後の分布からlogprobを生成する。
- additiveなHIP TokenSelect ABIと`sllm-core` semantic contractを追加した。Qwen/Gemmaのterminal projectionと同じqueueでselectorを
  submitし、固定16-byte selected recordだけをreadbackする。grammar mask、bias、presence/frequency、DRYはbounded F32/U8 inputへ
  lowerし、GPU selector非対応の高度なsamplerはhost pathへ明示的に残す。MTP block selectorは対象外である。
- GPU selectorのkernel/bridgeではtarget、capability、status、reserved、token範囲、finite logprobをfail-closedに検証し、通常の
  Argmax経路と既存public ABIを壊さない。GPU route選択後のCPU silent fallbackは行わない。
- integration reviewで、host samplerがeffective logit降順、初版GPU selectorがtoken ID昇順で累積していた差を検出した。legacy順を正に、
  f32 ordered-key探索とdouble massでGPUを一致させ、3-token反例と両実機matrixで固定した。
- 実モデルstructured generationでshared NFA rule endが複数callerへ混線する問題を検出した。明示Call/Return stackへ修正し、JSON Schemaの
  property orderを入力順で保持するため`serde_json/preserve_order`を有効化した。`{ {`回帰fixtureとQwen/Gemma実生成で再確認した。

## Verification status

| lane | 記録 | 判定 |
| --- | --- | --- |
| Host | workspace all-target tests、clippy `-D warnings`、fmt、Python API fixture、markdown link、diff check | PASS |
| HIP host/ABI | TokenSelect C ABI、Rust bridge、negative validation、selected-record contract、host CTest 4/4 | PASS |
| V620 `gfx1030` | UUID `GPU-76a08c...`、vocabulary `1,3,17,255,256,257,248320`×counter `0,1`、odd mask/additive、固定seed反復、CPU token exact/logprob tolerance `.005`、all-mask/nonfinite、fallback 0、selected record D2H 16 bytes/full-vocabulary D2H 0 | PASS（selector contract scope） |
| R9700 `gfx1201` | UUID `GPU-a8e9...`、V620と同一matrix、CPU token exact/logprob tolerance `.005`、all-mask/nonfinite、fallback 0、selected record D2H 16 bytes/full-vocabulary D2H 0 | PASS（selector contract scope） |
| MI300X `gfx942:sramecc+:xnack-` | wave64 feature-pinned compile/route | compile-only PASS、real selector correctness/performance deferred |
| Model integration | V620 `gfx1030`でQwen BF16/Gemma mixed NVFP4を実行。fixed-seed sampling、selected logprobs、JSON Schema、Qwen `n=2/8`、HIP-only/fallback 0、cleanup 0 | PASS |

GPU unavailable、timeout、crash、zero selectionはPASSとして数えない。V620/R9700 matrixはselector contract scope、model integrationは
V620実機scopeである。MI300X実機証跡はVM再確保後の別runへ引き渡す。

## 引き渡し

- gfx942はcompile/route evidenceを残し、MI300X real correctness/performanceをclaimしない。
- GPU selectorはcorrectness-firstのsingle-work-item/ordered-key実装であり、性能最適化はMI300Xを含むfuture profilingで判断する。

[archive plan](../../../../plans/archive/2026/08/21-31/phase40-token-selection-grammar-structured-generation.md) /
[GPU selector summary](../../../../../ci/matrix/phase40-token-selector-gpu-summary-v1.json) /
[main plan](../../../../plans/main-plan.md) /
[OpenAI compatibility](../../../../api/openai-compatibility.md) /
[runtime architecture](../../../../architecture/runtime.md) /
[GPU compatibility](../../../../compatibility/gpu.md) /
[AMD GPU compatibility](../../../../compatibility/amd-gpu.md)
