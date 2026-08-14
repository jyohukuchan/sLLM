# Phase 10: model本体FP8 W8A8

> 状態: complete
> 完了日: 2026-08-14

## 目的と結論

Phase 9のdtype非依存prepared execution/provider境界を再利用し、Qwen3.5-4Bのmodel本体を
OCP E4M3FNへ量子化して実行する。exact `gfx1201`にはnative hipBLASLt provider、exact `gfx1030`には
byte decode emulationとload時BF16 conversionを別providerとして実装した。

R9700 native FP8は4Bのmodel resident VRAMをBF16比約42.4%削減したが、32/32ではBF16よりprefill、decode、
E2Eのすべてが遅かった。このためFP8をdefaultへ昇格せず、明示opt-inとした。V620のemulationも
correctness providerに留め、実用的なfull-model pathは明示`converted-bf16`とする。

CDNA3 FNUZとexact `gfx942` providerはPhase 11、MI300X実機証拠はPhase 12で扱う。

## 固定した数値・model契約

- 正本はverified `Qwen/Qwen3.5-4B` BF16 lockから生成するreproducible sidecarである。
- text-linear 248 tensorをper-output-row FP32 scale付きOCP E4M3FNへ変換する。検討時のblock 128候補は、
  ROCm 7.14 hipBLASLtが実shapeでouter-vector scaleを提供するため採用しなかった。
- converterはRNE、finite saturation、特殊値規則を固定し、source lock、tool commit/hash、tensor range/hash、
  完全artifact hashをmanifestへ保存する。loaderはすべてをfail-closedに検査する。
- linear入力は実行時にper-row E4M3FNへ量子化し、weight/activation FP8、FP32 accumulation、BF16 outputとする。
  RMSNorm、softmax、RoPE、GDN state、KV cache、samplingは既存dtypeを維持する。
- FP8 valueとscaleはmodel residentに一度だけ配置し、activation quant workspaceとhipBLASLt solutionは
  prepare時に所有する。requestごとのweight変換やrepackは行わない。

## target別provider

| exact target | provider | resident / execution | 状態 |
| --- | --- | --- | --- |
| `gfx1201` | `native` | OCP E4M3FN / hipBLASLt OCP W8A8 | opt-in production |
| `gfx1030` | `emulation` | OCP E4M3FN / byte-decode W8A8 | correctness-only |
| `gfx1030` | `converted-bf16` | load時BF16変換 / Phase 9 BF16 | explicit production |

providerとexact targetが一致しない場合、solutionがない場合、sidecarが不整合な場合はprepare/loadで失敗する。
実行時失敗をBF16成功へ読み替えない。CLI、server、request auditはencoding/providerを区別する。

## 受入結果

- format oracle: OCP E4M3FN全256 byte、127/128/129境界、zero/subnormal/finite最大/NaN/Inf、RNE、saturationをPASS。
- operator: R9700 `gfx1201` kernel id 5 native、V620 `gfx1030` kernel id 6 emulationでM=1/M=3、K=128、N=256を
  独立oracleに対してPASS。max relative errorは約0.00364、fallbackなし、cleanup 0。
- 4B graph: 248 text-linear consumerとsidecar rangeを一致させ、実際の4B native generationをPASS。
- 4B logits: 3入力でBF16とtop-1が全一致し、最大KLD 0.02394（gate 0.05）をPASS。
- generation/service: fixed generation、OpenAI non-stream、SSE、`/v1/models`をnative FP8 aliasでPASS。
- V620: 4B `converted-bf16` generationをPASSし、BF16出力tokenと一致。emulationのfull-model性能測定は、
  correctness-only providerを長時間走らせる価値がないためoperator証拠に限定した。
- 回帰: workspace全target test、clippy、format、host CTest 3/3をPASS。

## 性能判断

R9700 Qwen3.5-4B、32 prompt / 32 completionの一回限りの影響測定は次の通り。

| path | prefill tok/s | decode tok/s | TTFT | E2E | resident VRAM | peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase 10 FP8 native | 486.26 | 31.58 | 73.06 ms | 1065.72 ms | 4,847,029,760 B | 5,044,359,040 B |
| Phase 9 BF16 | 531.71 | 37.04 | 66.69 ms | 912.86 ms | 8,411,592,192 B | 8,608,921,472 B |

FP8はresident VRAMを3,564,562,432 byte、約42.4%削減した一方、prefill約8.5%、decode約14.7%、
E2E約16.7%低下した。固定llama.cpp commit `f5919bf458ef190468b5c329bb293f8a54a1e69c`の
exact-token baselineもsLLMより速い。llama.cppはこのsidecarを消費できないため、FP8同士の擬似的な比較や
再変換は行わず、既存の固定BF16 peerを参照した。

## スコープ外とhandoff

- CDNA3 FNUZ、MI300X実行、MI300A/MI325X。
- KV cache FP8/NVFP4、Weight NVFP4、BF16からのruntime自動量子化。
- FP8 FlashAttention 4-like、vision、MTP、MoE、multi-request、multi-GPU。
- 公式27B FP8の完全なproduct support。

Phase 11へexact `gfx942` compile/link、OCP→FNUZ数値変換、wave64 BF16、hipBLASLt FNUZ、
VMMなしの`contiguous-resident` KV providerを渡す。実機PASSと性能値はPhase 12で取得する。

## 関連資料

- [Phase 10 history](../../../../../history/2026/08/11-20/phase10-fp8-w8a8.md)
- [メイン計画](../../../main-plan.md)
- [AMD GPU互換性](../../../../../compatibility/amd-gpu.md)
- [model lock](../../../../../models/model-lock.md)
