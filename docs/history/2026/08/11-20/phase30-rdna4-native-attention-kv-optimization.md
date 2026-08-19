# Phase 30: RDNA4 native attention/KV hardware-path最適化

> 完了日: 2026-08-19
> 決定: exact gfx1201のnative FP8 readとwave providerを限定採用

## 結果

Qwen系generic causal/full attentionのsemantic op、opaque KV state、public ABI、KV encodingを変えず、exact `gfx1201`の
`query_count=1`と`query_count>=32`へtarget-scoped providerを追加した。QK reductionを256-thread LDS treeから
8 wave × 32 laneのshuffle treeへ変え、8 wave partialだけをLDSで合成した。key当たりのblock同期は約11回から3回へ減った。
FP8/NVFP4のE4M3FN readはgfx1201 compiler builtinへ置き換え、software bit manipulationを除いた。

`query_count=2..31`とexact `gfx1030`は既存baselineを維持する。model graph、frontend、service、KV layoutにtarget分岐を
追加せず、runtime dispatch metadataとactual launchが同じtarget/M境界を使う。

## Candidate別判断

### C1 native FP8 codec

- read: **採用**。全256 E4M3FN code（NaN 2 codeを含む）がsoftware contractと一致し、mismatch 0、fallbackなしだった。
  gfx1201 code objectのactual attention kernelに`v_cvt_f32_fp8`が存在し、gfx1030には混入しない。
- append encode: **棄却**。chunk 1/32/256のpaired測定で256 tokenが68.69%悪化した。255/257の局所改善では
  adoption scope全体を満たさないため、KV appendと`kv_state_kernel`のsoftware encoderを維持した。
- native read単独のwave-only比短縮はprefix 3/255/256/257で4.60/4.75/15.62/13.50%、
  KV 1023/1024/1025/8193で14.02/13.81/13.45/13.93%だった。

### C2 wave-tiled online attention

FP16 decodeのbaseline比device p50短縮はprefix 3/255/256/257で6.12/13.86/16.36/7.22%、
KV 1023/1024/1025/8193で16.75/17.16/16.99/11.91%だった。FP8は同順に
0.64/24.92/26.94/26.67%、27.46/25.95/27.43/27.91%だった。prefill `M≈255`はFP16約21.0%、
FP8約31.5%短縮した。小さいprefill `M=17`には軽微な悪化があったため、production境界を事前に`M>=32`へ固定し、
2〜31をbaselineへ戻した。

### C3 prefill matrix attention

**negative/deferred**。llama.cppのRDNA向け`fattn-mma-f16.cuh`/`fattn-tile.cuh`とinstalled rocWMMAを比較したが、
既存sLLM kernelへ小さく移植できるproviderではなかった。tile layout、causal mask、GQA、online softmax、Qのdtype整合を含む
別のFlashAttention providerが必要で、Phase 30のbounded N0/N1 work unitを越える。matrix ISAを含まない現candidateへ
matrix名称を付けず、外部sourceもcopyしなかった。

## Correctness・数値

R9700 exact `gfx1201`とV620 exact `gfx1030`でFP16/FP8を各17 case実行し、全caseで全出力一致、最大絶対誤差0、
fallback 0、cleanup failure 0だった。gfx1201 native FP8 decode probeは256/256 code PASSした。

native FP8 readはbit-exactなN0である。wave providerはreal-number equation、入力集合、FP32 accumulator、softmax stage、
BF16 RNE outputを変えず、QKの固定balanced treeを同じ最長8段のwave treeへ変えるためN1とした。測定したfull-modelの
token recordは全てbaselineと一致したが、将来のrounding差を隠さず数値台帳へ登録した。

## Full-model performance

modelはQwen3.5-4B BF16 GGUF（SHA-256 `c571c54e...`）、ROCm 7.14.0、R9700 exact gfx1201で、
3 warmup + 10 measuredを各processに使用した。4108 input、output 32をbaseline/candidate各3 independent processで
counterbalanceしたprocess中央値は次の通りである。

| 指標 | baseline | candidate | 改善 |
| --- | ---: | ---: | ---: |
| TTFT | 6.856294 s | 6.198135 s | 9.60% |
| prefill | 6.771907 s | 6.113465 s | 9.72% |
| E2E | 8.259185 s | 7.502338 s | 9.16% |
| decode | 16.8102 tok/s | 18.1321 tok/s | 7.86% |

267 input controlはTTFT/prefill/E2E/decodeが1.89/2.10/1.92/1.72%改善した。29 input controlは
TTFT/prefillが-0.60/-0.46%、E2E/decodeが+0.24/+0.35%で、1 processのsub-1% noiseとしてstable悪化に数えなかった。
全runのtoken recordは一致した。

10000+ inputはpreflightで53,758,880,592 byteを要求し、available 34,135,343,104 byteを超えたためfail-closedした。
kernel crash/fallbackではなく現行full-prefill workspace制約であり、性能PASSへ数えない。FP8-KV full-model routingを追加する
public CLI変更もPhase 30では行わず、FP8の採用証拠はoperator scopeに限定した。

## Architecture・resource

- workgroup 256、grid、opaque KV value/scale plane、transaction/publication、scratch allocationを維持した。
- wave providerのLDSは既存256 float reduction bufferを再利用し、追加allocation、host readback、device-wide synchronizeはない。
- actual dispatch symbolはgfx1201 wave providerを明示し、gfx1030/complementはbaseline symbolを報告する。
- final sourceから実験用disable/force-baseline compile routingを除去し、棄却したnative append candidateも残していない。

## Verification

- exact gfx1030/gfx1201 CMake build: PASS。
- `sllm-core` execution tests 61/61、`sllm-hip` library tests 95/95: PASS。
- gfx1030/gfx1201 FP16/FP8 attention oracle各17/17: PASS。
- gfx1201 FP8 decode 256-code probe: PASS、fallback false。
- full-model 4108 input、各3 process counterbalanced: adoption threshold PASS。
- bounded summary/schema/test、markdown/link、license/provenance、workspace checks: PASS。

集約値とidentityは[Phase 30 bounded summary](../../../../../ci/matrix/phase30-rdna4-attention-kv-summary-v1.json)を正とする。

[Phase 30計画](../../../../plans/archive/2026/08/11-20/phase30-rdna4-native-attention-kv-optimization.md)
[数値・出力影響変更台帳](../../../../compatibility/numerical-output-changes.md)
[メイン計画](../../../../plans/main-plan.md)
