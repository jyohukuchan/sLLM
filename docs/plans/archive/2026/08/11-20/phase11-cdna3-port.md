# Phase 11: FP8/BF16のCDNA3移植

> 状態: completed
> 作成日: 2026-08-14

## 完了結果

- exact `gfx942` / Code Object V6 / wave64 / `xnack=off` / `sramecc=on`でROCm 7.14.0 native runtimeをcompile/linkした。
- 全256 byteのOCP E4M3FN→E4M3FNUZ数値変換、FNUZ dynamic activation量子化、hipBLASLt FNUZ provider、
  model-resident load変換を実装した。OCP payloadのraw reinterpretやsilent BF16 fallbackは行わない。
- M=1 BF16 matmulとRMSNormをgfx942専用wave64 kernel ID/symbolへ分離し、wave32固定箇所をcompile auditした。
- HIP VMM capability=falseでは通常のdevice allocationを使う`contiguous-resident` KVをprepare時に選択し、
  capability=trueでは既存virtual-contiguous vAttentionを維持する。fake capability host testで両分岐を確認した。
- production CLI/serverはexact gfx942で`native-fnuz`を選び、同じQwen graph、semantic operation、service、
  transactional state契約を使う。
- MI300X候補manifestとpreflight/operator/slice/full-model/service/performanceのdry-run runnerを追加した。
  MI300X実機実行、FNUZ solution/numerical PASS、性能値はPhase 12の開始条件として残す。

## 目的

Phase 9のBF16 pathとPhase 10のFP8 W8A8 pathを、exact `gfx942`、wave64、CDNA3 native FNUZ FP8へ移植する。
Phase 11は実装とlocal compile/oracleを完成させ、Hot Aisle MI300X VMで即座に実行できるcandidateを作る。
実機evidenceの取得と性能判断はPhase 12とし、Phase 11完了を未所有GPUの実行でblockしない。

## CDNA3で変わる契約

- binary targetはexact `gfx942`とし、初期FP8 fast pathに`gfx9-4-generic`を使わない。
- MI300Xはwave64であり、local RDNA2/RDNA4のwave32前提を持ち込まない。
- model storageのOCP E4M3FNをVRAM load時にE4M3FNUZへ数値変換する。専用のFNUZ量子化modelは作らない。
- hipBLASLt FP8 providerはFNUZ data type、solution support、scale/accumulation契約をshapeごとに確認する。
- AMDの公開MI300X llama.cpp例では`gfx942:sramecc+:xnack-`、wave64、VMMなしと報告される。したがって
  HIP VMM vAttentionだけをproduction KVの必須条件にせず、同じopaque KV契約を使う
  `contiguous-resident` providerを追加する。

## KV memoryの方針

`contiguous-resident`はPaged Attentionへの切替ではない。token-major FP16 K/V layout、連続device pointer、
attention ABI、request ownershipは現行vAttentionと同じに保ち、logical capacity分を通常のdevice allocationで
確保する。capability queryでVMMが使えるtargetは既存vAttention、使えない`gfx942`はcontiguous-residentを
prepare時に明示選択する。実行時エラー後のsilent fallbackは行わない。

MI300Xの192 GiB HBMでは4B/9Bおよび限定27B spotの単一request KVを十分収容できる。Phase 11でmodel graphから
layer/head/dtype/context別の必要byteを計算し、capacity超過をallocation前に診断する。

## スコープ外

- MI300X実機PASS、性能値、software tupleの`project-verified`昇格。これらはPhase 12。
- MI300A/MI325XのSKU検証、`gfx942`から全CDNA3 SKUへの無条件一般化。
- multi-GPU、Infinity Fabric、RCCL、P2P/RDMA、partition modeの対応。
- Paged Attention、prefix sharing、continuous batching、KV量子化。
- CDNA4 `gfx950`、generic code object、FP8 FlashAttention 4-like。

## 受入条件

1. `gfx942` target別artifactをROCm 7.14.0でcompile/linkでき、artifact metadataへexact targetとcodegen featureを
   保持する。wave32-only builtin、lane mask、warp定数をcompile auditで列挙する。
2. E4M3FN→E4M3FNUZは全256 byte patternと境界値を独立oracleで検査し、NaN/Inf、zero、subnormal、finite範囲、
   saturationの意味を記録する。raw reinterpretを禁止する。
3. FP8 hipBLASLt providerはFNUZ typeとscale/accumulationを使用し、support queryが失敗するshapeをprepare時に
   明示拒否または別の明示providerへ振り分ける。
4. BF16 MMVF/GEMM、GDN、RMSNorm/fusion、FA2-style attention、RoPE/KV appendをwave64で成立させる。
   `gfx1030`用state layoutや32-lane reductionを暗黙に再利用しない。
5. VMM capability=falseでcontiguous-resident KVを選び、1023/1024/1025、capacity境界、cancel/error/drop、
   request間再利用、cleanupをhost contractとGPU probeで検査できる。VMM unavailableはPASSでもfallbackでもなく、
   選択済みcapability pathとして記録する。
6. 4B BF16/FP8のproduction graphが同じRust service、semantic op、completion、transactional state契約を使う。
7. MI300X用preflight、operator、slice、full-model、service、performanceのrunnerをVM取得前にdry-runできる。
   model、binary、raw traceをrepositoryへ追跡しない。
8. affected host test、`gfx942` compile-only、local GPU非回帰、1回のintegration review、指摘箇所のfocused
   re-review、provenance、runtime/compatibility/main plan/history同期を完了する。

## 実装順序

### P11-A0: gfx942 readiness audit

- native kernels、inline asm、wave reduction、launch bounds、LDS、atomic、codegen feature条件を棚卸しする。
- exact `gfx942` compile matrixを追加し、`sramecc`/`xnack` suffixを捨てないartifact keyを確認する。
- Hot Aisle向けpreflightでOS/kernel/driver/ROCm、BDF/UUID、gfx、wave、CU、HBM、ECC、partition、VMM、
  hipBLASLt/rocprofiler、root権限、loader pathを収集する。

### P11-A1: OCP E4M3FNからFNUZへのload変換

- scalar/reference converterを先に作り、全byte patternとblock scaleを検査する。
- vectorized device converterをmodel load pathへ追加し、変換後weightをmodel-residentに保持する。
- 変換前後hash、元model lock fingerprint、resident encoding、変換時間/一時workspaceを監査可能にする。

### P11-A2: CDNA3 FP8 provider

- hipBLASLt FNUZ providerをM=1/M>1、Qwen実shape、非整列/境界shapeでquery・benchmarkする。
- dynamic activation quantizationもFNUZへ変換し、weight/activation/scale/FP32 accumulation/BF16 outputを
  operator contractへ固定する。
- support solutionがないshapeだけcustom FNUZ kernel候補を検討し、generic/BF16へのsilent fallbackはしない。

### P11-A3: BF16 wave64 port

- Phase 9 MMVF v3、GDN state、prefill provider、fusion、completion segmentをwave64対応させる。
- reduction lane、state layout、occupancy/LDSを`gfx942`専用providerとして分離し、RDNA providerを汚さない。
- compile-onlyとhost fixtureでprovider selection、shape、workspace、cleanup contractを固定する。

### P11-A4: contiguous-resident KVとattention

- 既存opaque KV providerへ`contiguous-resident`を追加し、capacity byte計算、allocation、append、read、rollback、
  terminal ownershipを実装する。
- VMM capability=true/falseの両selectionをfake capability fixtureで検査し、local RDNA vAttention非回帰を行う。
- exact `gfx942` FA2-style attentionをcompileし、wave64 reductionと連続K/V pointerの数値probeを用意する。

### P11-A5: production統合candidate

- 4B BF16/FP8についてmodel load→prepare→prefill→decode→sampling→serviceまでのrunnerを組み立てる。
- operator/sliceのNumPy fixture、固定generation、OpenAI non-stream/SSE、cancel/cleanupを一つのmanifestから選べる
  ようにする。Phase 12の実行順をrunner profileへ固定する。
- 公式27B FP8 interop spotはmemory/model-format対応を確認し、未対応model差は4B pathのblockerと分離する。

### P11-A6: local統合とPhase 12 handoff

- H0〜H2、`gfx942` compile-only、V620/R9700 affected GPU非回帰を行う。
- candidate source/build inputs、toolchain、model locks、artifact digest、expected capabilityをmanifestへ固定する。
- Phase 12のfirst-hour stop/go条件、実行case、時間上限、証拠保存先を確認してplanをarchiveへ移す。

## Phase 12への開始条件

- `gfx942` artifact、4B BF16/FP8 model lock、runner、NumPy oracle、report schemaがlocalで準備済み。
- VMM=falseを正常なcapability branchとして扱うcontiguous-resident KVが実装済み。
- VM上でsource設計や広いtest作成を始めず、失敗をlocalで再現・修正できる最小probeが揃っている。
- model/binaryをVMへ転送またはVM側で検証取得する手順が決まり、download待ちで課金時間を浪費しない。

## 終了時更新先

- [Phase 12 archive](phase12-mi300x-validation.md)
- [メイン計画](../../../../main-plan.md)
- [AMD GPU互換性](../../../../../compatibility/amd-gpu.md)
- [software互換性](../../../../../compatibility/software.md)
- [runtime architecture](../../../../../architecture/runtime.md)
- [Phase 11 history](../../../../../history/2026/08/11-20/phase11-cdna3-port.md)
