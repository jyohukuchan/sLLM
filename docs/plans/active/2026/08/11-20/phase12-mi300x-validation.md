# Phase 12: Hot Aisle MI300X単体実機確認

> 状態: ready
> 作成日: 2026-08-14

## Phase 11 handoff

- Phase 11はexact gfx942 native compile/link、FNUZ host oracle、wave64 BF16 provider、contiguous-resident KV、
  production `native-fnuz`統合を完了した。
- `python3 ci/tools/run_phase11_mi300x_candidate.py --dry-run`で、全6 profileと所要時間見積りをVM取得前に検査する。
- local candidateはMI300X実行をclaimしない。最初の実行主張は本Phaseのpreflight/operator reportから開始する。

## 目的

Phase 11から引き継いだlatest main candidateをHot AisleのMI300X x1 VMで実行し、exact `gfx942`のBF16、FNUZ FP8、wave64、
contiguous-resident KV、full model、OpenAI-compatible service、性能とcleanupをfail-closedに確認する。
VM固有tupleの証拠と、exact gfx942 kernelに一般化できる証拠を分けて記録する。

## MI300Xを管理できない期間の扱い

Hot Aisle VMを継続管理できる時間が確保できるまでは本Phaseを`ready`で保持し、VMを作成・起動しない。
その間は[ローカル先行実行キュー](phase12-wait-local-forward-queue.md)に従ってPhase 13以降を進めてよい。
これは本Phaseの完了、skip、順序変更を意味しない。

先行変更後に本Phaseを開始する際は、その時点の最新mainからexact `gfx942` artifactを再buildし、dry-run、
source/build identity、runner/report schemaを再確認する。Phase 12のmatrixはQwen3.5 4B/9B BF16/FP8、
contiguous-resident KV、service、性能比較のまま維持し、Gemma、NVFP4、KV量子化、MoEを自動追加しない。

## hardware判断

Hot Aisle Small VMのMI300X x1でPhase 12には十分である。公開仕様は192 GB HBM3、8または13 CPU core、
224 GB system RAM、12 TB NVMeであり、sLLMの現在のsingle GPU/batch 1、4B/9B BF16・FP8、限定27B FP8 spot、
operator/full-model/service/performanceを一台で実行できる。可能なら同料金の13 CPU core variantを選ぶ。

一台では次を検証できないため、Phase 12の主張に含めない。

- multi-GPU、Infinity Fabric、P2P、RCCL、RDMA/RoCE、tensor parallel。
- 8 GPU bare metal固有のNUMA、partition、firmware/BMC、node間network挙動。
- MI300A/MI325XのSKU互換性、別cloud/bare-metalでの性能同等性。
- VMのvirtualization方式の影響を除いたMI300X hardware peak。性能値はHot Aisle VMの完全tupleに限定する。

## 利用時間の目安

clean candidateなら合計10〜12 GPU時間を見込む。連続で借りっぱなしにせず、minute billingを利用して二回へ
分ける。事前に12 GPU時間を標準予定、16 GPU時間を現実的な上限、環境依存問題が残る場合だけ追加4時間を
別日へ回す。

| session | 内容 | 目安 |
| --- | --- | ---: |
| 1 | provision、identity/health/VMM/hipBLASLt/profiler preflight、artifact起動、operator smoke | 2〜3時間 |
| local修正 | VM停止後にcompile/oracle/runnerを修正 | 課金なし |
| 2 | operator数値、4B BF16/FP8、9B spot、service、性能、llama.cpp比較 | 6〜8時間 |
| 予備 | immutable candidateのfocused rerunまたは公式27B FP8 spot | 0〜4時間 |

Hot Aisleの2026-08-14時点の新規顧客料金は$2.99/GPU/hourでminute billingであるため、12時間は約$36、
16時間は約$48、20時間でも約$60が概算となる。これは予算gateではなく、VMを停止する判断の目安である。

## VM取得前の準備

- Phase 11から引き継いだlatest main candidateのsource/build identity、`gfx942` artifact、runner/report schemaを固定する。
- 4B BF16/FP8のverified model lockを用意し、9Bと公式27B FP8は必要なfileだけを選ぶ。
- model cacheを圧縮・hash検証可能な形で転送するか、VM作成直後に並列取得する。モデル取得中にcompile設計を
  始めない。model/binary/raw profileをGitへ追加しない。
- Ubuntu 24.04を選択できる場合は選ぶ。provider imageのkernel/driver/ROCmを推測せず、最初に完全tupleを記録する。
- project標準はROCm build/runtime 7.14.0である。VM imageが異なる場合はfull rootを使って同releaseの
  self-contained user-space rootを用意できるか確認するが、provider管理driverを無断で交換しない。
- fixed llama.cpp source commit、build引数、同じ4B BF16/GGUF変換lock、benchmark prompt/token条件を準備する。

## first-hour stop/go

最初の60分で次を確認し、重大な不一致があれば長いmodel runを行わずVMを停止する。

1. 単一visible deviceがMI300X、exact `gfx942`、wave64、約192 GiB HBM、期待するCU数として見える。
2. BDF/UUID、`sramecc`/`xnack`、compute partition、VM/SR-IOV、ECC、temperature/power/clock、process ownershipを
   取得できる。telemetry unavailable fieldは明示し、zeroとして扱わない。
3. build/runtime ROCm、compiler、HIP/ROCr/hipBLASLt absolute pathが同じrelease/root契約を満たす。
4. `hipDeviceAttributeVirtualMemoryManagementSupported`を実測する。falseは想定内であり、
   contiguous-resident providerを選択できれば続行する。trueでも自動的にvAttentionへ切り替えず、candidateの
   固定providerを守る。
5. hipBLASLt FNUZ solution query、tiny BF16/FP8 kernel、event、allocation/copy、rocprofiler権限が実行できる。
6. artifactがexact targetでloadでき、CPU fallback、generic code object、別ROCm rootを使わない。

## 実行順序

### P12-A0: provisionとidentity

- 13 CPU core Small VMを優先し、公開IP/firewall/SSH keyを最小範囲で設定する。
- preflight reportを取得し、provider image、kernel/driver、ROCm root、GPU identity、partition/VMMを固定する。
- foreign processまたはGPU共有が疑われる場合は性能runを行わず、support確認または別VMへ切り替える。

### P12-A1: model-free/operator correctness

- G0/G1、BF16 MMVF/GEMM、FP8 FN→FNUZ converter、FNUZ matmul、GDN、RMSNorm/fusion、RoPE、KV append、
  FA2-style attentionを数値oracleと比較する。
- 非整列値、127/128/129、255/256/257、1023/1024/1025とQwen実shapeを含める。
- timeout、crash、zero selection、CPU execution、unsupported solutionをPASSにしない。

### P12-A2: 4B/9B model統合

- 4B BF16と派生FP8を同じprompt/token条件でslice、logits、fixed/Unicode/stop generationへ通す。
- top-1/KLDとselected provider/encodingを記録し、FP8のFNUZ resident変換を監査する。
- 9B BF16/FP8はarchitecture/shape coverageのspotに限定し、4Bと同じ全matrixを繰り返さない。
- 公式27B FP8は4BがPASSし時間が残る場合だけload、最小slice、短いgenerationを行う。未対応model差を
  CDNA3 kernel defectと混ぜない。

### P12-A3: KV/service/cleanup

- VMMなしのcontiguous-resident KVでshort/long capacity、cancel/error/drop、連続request、VRAM復帰を確認する。
- OpenAI Chat Completionsのraw non-stream/SSE、OpenAI client、reasoning split、disconnect/recoveryをsmokeする。
- request/model/workspace allocation、GPU process、ECC/healthをpre/during/postで記録し、終了後にzero/基準へ戻す。

### P12-A4: performanceと比較

- direct engineを正本に、4B BF16/FP8のshort-odd、32/32、prefill-long、decode-longをwarmup 3回、計測10回で
  実行する。VM noiseとclock/temperature/powerを同時記録する。
- fixed llama.cppを同じVM、model revision、入力/出力token、dtype条件で実行し、TTFT、prefill/decode token/s、
  TPOT、E2E、resident/peak VRAMを比較する。
- sLLM FP8とllama.cpp BF16/別quantを同等比較と呼ばない。FP8はsLLM BF16比も併記する。
- raw traceは代表caseだけ取得し、VM外のlocal artifact storeへ退避する。常時計測overheadを性能値へ混ぜない。

### P12-A5: closeout

- immutable final candidateで、変更したfindingだけをfocused rerunする。全matrixの再実行を既定にしない。
- report/hash/summaryを退避し、model cache、credential、SSH key、VM diskを確認してVMを破棄する。
- compatibility evidenceをexact Hot Aisle VM tupleへ限定して追記し、Phase 12 planをarchiveへ移す。

## 受入条件

1. exact `gfx942` artifact、wave64、FNUZ FP8/BF16、contiguous-resident KVがCPU/generic/silent fallbackなしで実行される。
2. operator oracle、4B slice/generation、KV/service/cleanupがPASSし、9B spotを少なくともBF16で確認する。
3. VMM、partition、VM形態、OS/kernel/driver/ROCm/library、GPU identity、health/ECC、artifact/model lockをreportする。
4. performanceをVM tuple限定のrepeated median/dispersionで残し、fixed llama.cppとの差とsLLM BF16↔FP8差を分ける。
5. MI300A/MI325X、multi-GPU、bare metalへ結果を一般化しない。
6. 1回のintegration reviewと指摘箇所だけのfocused re-reviewを完了し、main plan、runtime、GPU/software互換性、
   Phase 12 historyを同期する。

## 公式資料

- [Hot Aisle pricing](https://hotaisle.xyz/pricing) — Small VM仕様、minute billing、現行料金。
- [AMD GPU specifications](https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html) — MI300Xの`gfx942`、
  192 GiB、304 CU、wave64。
- [AMD MI300X platform data sheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-platform-data-sheet.pdf) — HBM3、CDNA3、FP8の製品仕様。
- [ROCm llama.cpp example](https://rocm.docs.amd.com/projects/llama-cpp/en/latest/examples/llama-cpp-examples.html) —
  MI300Xの公開実行例とVMMなし/wave64の観測。
- [ROCm 7.14.0 release notes](https://rocm.docs.amd.com/en/docs-7.14.0/about/release-notes.html) — MI300Xとprofilerのrelease情報。

## 終了時更新先

- [メイン計画](../../../../main-plan.md)
- [GPU互換性](../../../../../compatibility/gpu.md)
- [AMD GPU互換性](../../../../../compatibility/amd-gpu.md)
- [software互換性](../../../../../compatibility/software.md)
- [runtime architecture](../../../../../architecture/runtime.md)
- [Phase 12 history](../../../../../history/2026/08/11-20/phase12-mi300x-validation.md)
