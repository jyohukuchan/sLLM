# Phase 16 KV cache FP8/NVFP4履歴

## 2026-08-16: 詳細計画作成

- 残タスクの依存関係を見直し、FP8 KVを先に、NVFP4 KVを次に実装する順序を維持した。
- Phase 16Fのprimary artifact `unsloth/gemma-4-12b-it-NVFP4`がmixed recipeでFP8 KVを要求するため、
  first-class FP4 full-model integrationよりPhase 16を先に完了する順序へ固定した。
- Phase 6のopaque KV、VMM virtual-contiguous、Phase 11のcontiguous-resident、Phase 13のtransactionを維持し、
  value/scale planeだけをversioned encodingとして追加する計画とした。
- append時の一度だけの量子化、attentionからの直接消費、全cache FP16/BF16 mirror禁止、K/V atomic publication、
  cancel/recovery、quality/memory/performanceの受入条件を固定した。
- canonical runtime matrixはexact `gfx1030`/`gfx1201`とし、利用可能な実機がない`gfx942`はcompile/host contractを
  超えてPASSとしない。本時点ではsource、ABI、kernel、model artifactを変更していない。

[対応する計画](../../../../plans/active/2026/08/11-20/phase16-kv-cache-fp8-nvfp4.md)
