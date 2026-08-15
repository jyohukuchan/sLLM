# AMD GPU互換性方針

> 最終更新: 2026-08-16
>
> この文書はAMD向けの識別規則と初期候補を記録する。現時点の初期targetはすべて`lifecycle=experimental`である。計画targetのevidenceは`unverified`、canonical local実機のformal model-free G0/G1とPhase 6 A0 HIP VMM PoCは検証した限定範囲だけ`project-verified`とする。

共通の状態、resource gate候補、NVIDIAを含む将来例は[GPU互換性方針](gpu.md)を参照する。

## 正規識別子

AMD GPUの正規キーは製品名や「RDNA 2」「CDNA 3」ではなく、HIPが実行時に返す`gcnArchName`と、それに一致するコードtargetである。[`hipGetDeviceProperties`](https://rocm.docs.amd.com/projects/HIP/en/docs-7.14.0/reference/hip_runtime_api/modules/device_management.html#_CPPv422hipGetDevicePropertiesP15hipDeviceProp_ti)はdevice propertiesを取得し、その`gcnArchName`はAMD GCN architecture nameを返す。RDNA、CDNA、Radeon、Instinctなどは表示・説明用の分類に留める。

AMD向け`binary_key`は少なくとも次の要素を持つ。

```text
(backend=HIP,
 execution_abi=(amdhsa, code_object_version),
 code_target=<exact gfx target または generic processor>,
 generic_processor_version=<0 または 1..255>,
 codegen_features=(xnack, sramecc, wavefront size, ...))
```

exact code objectでは`generic_processor_version=0`、generic code objectでは`1..255`とする。[LLVMのgeneric processor versioning](https://llvm.org/docs/AMDGPUUsage.html#generic-processor-versioning)は、generic targetへprocessorが追加される際に互換codegenが変わり得るためversionを上げ、code objectのversionがprocessor追加時のversion以上ならロード可能と定義している。

正規化時は次を守る。

- `gcnArchName`からexact target（例: `gfx942`）とfeature stateを失わない。marketing SKUから推測した値だけでbinaryを選ばない。
- wave32/wave64はkernelの同期・resource前提に関係する。codegen側のwavefront sizeを`binary_key`、実行deviceとkernel側の対応を`capability_profile`へ記録する。
- matrix engine、precision、wave size、memory、帯域、LDS/registerなどは`capability_profile`へ分離する。同じ`gfx` targetでもSKUごとのresource gate結果は異なり得る。

## `xnack`と`sramecc`のstate

`xnack`と`sramecc`はbooleanや不明値に潰さず、LLVMの[AMDGPU ELF feature state](https://llvm.org/docs/AMDGPUUsage.html#amdgpu-elf-header-e-flags-table-v4-v5)に合わせて正規化する。

| State | 意味 |
| --- | --- |
| `unsupported` | processorがそのfeatureを実装しない |
| `any` | featureを実装するprocessorで、実行時設定がoff/onのどちらでもロード可能なcode object |
| `off` | featureを実装し、無効状態を要求するcode object |
| `on` | featureを実装し、有効状態を要求するcode object |

`any`は「不明」や「検査省略」ではない。off/onのどちらでも実行できるよう生成されたことを表す。`off`/`on`はdeviceの実行時設定と一致を要求する。feature stateは単なる性能メタデータではなく、loader compatibilityに関わる`binary_key`の一部である。

## Capability情報の補完

HIP runtimeから直接取得できる値だけではprofileは完成しない。

- `gcnArchName`、warp/wave、memory量などはHIP runtimeから取得する。
- matrix coreとprecisionのhardware evidence、LDS/register等はexact targetとvendor device IDをキーに、確認した資料version付きmappingで補完する。AMDの[GPU specifications](https://rocm.docs.amd.com/en/docs-7.14.0/reference/gpu-specs.html)を初期の一次資料とする。
- library pathは使用中のcomponentのqueryを優先する。ROCm 7.14.0のhipBLASLt 1.4.1では`isSolutionSupported()`も利用候補だが、query結果はそのlibrary/problemの証拠に限定する。
- 文書やlibrary queryで確定できない場合は、対象命令またはkernelのcapability probeを行い、exact target、driver/runtime/library versionとともに記録する。

帯域とmemoryのeligibility/admission上の扱いは共通文書のResource gate候補に従う。

## Generic code objectとexact fast path

[LLVM AMDGPU backendのprocessor表](https://llvm.org/docs/AMDGPUUsage.html#processors)は、generic processorが実行できるexact processorの集合と、generic化に伴う制限を定義している。[AMDのcode portability資料](https://rocm.docs.amd.com/projects/llvm-project/en/docs-6.4.2/conceptual/code-portability.html)も、`gfx10-3-generic`が`gfx1030`–`gfx1036`向けbinaryを一つにまとめられる一方、lowest-common-denominatorによる性能影響があり得ると説明している。

sLLMは次の二経路を分ける。

- generic path: 配布binary数を抑えるbaseline。対象processor集合、`generic_processor_version`、feature stateを照合する。
- exact fast path: `gcnArchName`とfeature stateが一致するexact target向け。固有命令とtarget別resource tuningを利用する候補とする。

generic processor versionをELF `e_flags`へ保持する仕組みは[Code Object V6以降](https://llvm.org/docs/AMDGPUUsage.html#amdgpu-elf-header-e-flags-table-v6-onwards)に定義されるため、sLLMのgeneric pathはCode Object V6以降に限定する。基準toolchainのROCm 7.14.0同梱compilerが意図したgeneric version/feature stateを生成し、同releaseのROCr loaderが初期対象実機で受理することは今後検証する。検証完了まではgeneric pathも`evidence=unverified`である。

generic binaryへ黙ってfallbackしてよいとは限らない。特に`gfx9-4-generic`は`gfx942`と`gfx950`を覆う一方、LLVM表ではFP8/BF8命令と変換命令が利用不可であるため、初期CDNA 3 FP8 pathには採用しない。

## 初期AMD target集合

次は初期実装の計画範囲であり、現在の動作実績ではない。

| 表示分類 | exact target | generic baseline | lifecycle | evidence |
| --- | --- | --- | --- | --- |
| RDNA 2 | `gfx1030`–`gfx1036` | `gfx10-3-generic`、Code Object V6+ | `experimental` | `unverified` |
| RDNA 4 | `gfx1200`, `gfx1201` | `gfx12-generic`、Code Object V6+ | `experimental` | `unverified` |
| CDNA 3 | `gfx942` | 初期FP8 pathでは使用しない | `experimental` | `project-verified`（下記Hot Aisle tuple/scope限定） |

[ROCm 7.14.0 release notes](https://rocm.docs.amd.com/en/docs-7.14.0/about/release-notes.html)は、ROCm componentと公式対象製品をexact target単位で掲載する。generic processorのcoverage、compilerがtargetを生成できること、製品・OS構成のvendor supportは別の事実である。

## FP8の根拠とinterop contract

FP8はhardware ISA evidenceとlibrary/sLLM contractを分ける。

### Hardware ISA evidence

AMDのROCm 7.14.0[data types and precision support](https://rocm.docs.amd.com/en/docs-7.14.0/reference/precision-support.html)は、CDNA 3 matrix coreのnative FP8をFNUZ、RDNA 4のFP8をOCP系として区別している。[gfx942 instruction syntax](https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/AMDGPU/AMDGPUAsmGFX940.html)にもFP8/BF8 matrix命令が列挙される。これらをhardware capability mappingの根拠とする。

- `gfx942`（CDNA 3）: native FNUZ系FP8（E4M3 FNUZ/E5M2 FNUZ）。
- `gfx1200`/`gfx1201`（RDNA 4）: OCP系FP8（E4M3/E5M2）。
- `gfx1030`–`gfx1036`（RDNA 2）: native FP8 matrix pathなし。将来FP8 modelを扱う場合は変換またはemulationを別pathとして明示する。

### sLLMとhipBLASLtの初期interop contract

[ROCm 7.14.0 release notes](https://rocm.docs.amd.com/en/docs-7.14.0/about/release-notes.html)は同releaseのhipBLASLtを1.4.1とする。そのversionの[hipBLASLt precision support](https://rocm.docs.amd.com/projects/hipBLASLt/en/docs-7.14.0/reference/data-type-support.html)は、同表がlibrary自身のdatatype supportでありhardware supportを示さないと明記している。

sLLMがhipBLASLt 1.4.1のFP8 GEMM pathを使用する場合の初期contract候補は次とする。

- exact `gfx942`: `hipblaslt_f8_fnuz`/`hipblaslt_bf8_fnuz`を使う。
- exact `gfx1200`/`gfx1201`: `hipblaslt_f8`/`hipblaslt_bf8`を使う。
- model storage encoding、sLLM kernel input、hipBLASLt datatypeが異なる場合は明示的に変換し、FNUZとOCP payloadを再解釈しない。

`gfx942`はPhase 11のexact compile/link/host oracleに加え、Phase 12のHot Aisle MI300X VFで実機検証した。`gfx1201`はPhase 10で実装・local実機検証済みだが、
hipBLASLt表だけをhardware全体の対応根拠にせず、exact target、runtime capability、library query、
shape/alignmentをすべて満たしたproblemだけdispatchする。

### Phase 10〜12の計画contract

- Phase 10はexact `gfx1201`でOCP E4M3FN native W8A8、exact `gfx1030`で明示的なW8A8 emulationと
  BF16 conversionを実装・検証した。RDNA2 pathをnative FP8と表記しない。R9700 nativeはVRAMを削減したが
  BF16より遅いため自動provider優先順位を上げず、V620 emulationとは別の内部evidence scopeとして記録する。
- Phase 11はexact `gfx942`、wave64、FNUZ FP8への実装、compile/link、host oracleを完了し、model storageの
  E4M3FNをload時にFNUZ residentへrebasingする。Phase 12ではHot Aisle MI300X VF x1、Ubuntu 24.04、
  kernel `6.8.0-124-generic`、amdgpu `6.16.13`、ROCm 7.14.0/HIP 7.14.60850、
  `gfx942:sramecc+:xnack-`、wave64、304 CUのtupleでoperator、4B/9B model、service、performanceを実機PASSした。
  `gfx9-4-generic`やOCP/FNUZ payloadのraw reinterpretを使わない。
- AMDの公開MI300X llama.cpp例は`gfx942:sramecc+:xnack-`、wave64、VMMなしを報告している。この情報は
  sLLMのHot Aisle VM実測ではないため`vendor-published observation`として扱い、Phase 12 preflightで
  `hipDeviceAttributeVirtualMemoryManagementSupported`を再取得する。
- VMMなしのtargetには、同じtoken-major FP16 K/Vとattention ABIを使う`contiguous-resident` KV providerを
  capabilityで選択する。Phase 12の比較条件を固定するためexact `gfx942`は実測VMM=trueでも同providerを明示選択し、
  1023/1024/1025 capacity、cancel/recovery、cleanup zeroを確認した。他のVMM対応targetのvAttentionは廃止せず、
  Paged Attentionへ暗黙に切り替えない。

### Low-bit modelのstatusとユーザーinterface

- `default`、`opt-in production`、`correctness-only opt-in`は過去のprovider/converter採否を説明する内部evidence表現であり、
  ユーザーへ別の起動command、許可flag、確認、通常警告を要求するcompatibility modeではない。
- 最終GGUF loaderはBF16、FP8、NVFP4、MXFP4を同じ操作で読み、artifact metadataとexact targetからverified providerを
  自動選択する。provider overrideは開発・benchmark用に限定し、通常利用では量子化artifactの選択を十分なユーザー意思とする。
- target別runtime成熟度、provider優先順位、sLLM製PTQ converter品質、提供元PTQ/QAT/native model evidenceを分離する。
  converterのKLD不採用をencoding/provider全体へ一般化せず、提供元低bit modelへ存在しないBF16比較を要求しない。
- packed-dequantを通常経路として選べてもRDNA2/RDNA4/CDNA3のnative FP4とは呼ばない。破損artifact、未対応encoding、
  実行不能targetは警告付きfallbackではなくerrorにする。

### Phase 15 Weight NVFP4

- exact `gfx1201`と`gfx1030`は同じ`packed-dequant` providerを実GPUで検証した。これはpacked E2M1を直接消費するが、
  native FP4 arithmetic、vendor library FP4 GEMM、W4A4を意味しない。
- M=1/M>1、K/N 15/16/17および31/32/33を含む7 caseを両targetで独立FP32 oracleへ照合し、fallbackなしでPASSした。
- Qwen3.5-2Bのtop-1は両targetで3/3一致したが最大KLD `0.2637523`が既定`0.05`を超えたため、このsLLM製PTQ artifactを
  推奨converter結果として採用しない。providerのdispatch correctnessと、別の提供元QAT/native artifactのmodel品質は独立に判定する。
- direct-engine follow-upではBF16比でresidentを52.43%削減した一方、NVFP4 decodeはV620で約21〜22%、R9700で
  約20〜22%低下し、R9700のprefill/TTFTは大幅に退行した。これは両target共通のpacked-dequant実装の結果であり、
  RDNA4 native FP4性能として一般化しない。
- exact `gfx942`はdescriptorとcompile-only対象に留め、実行、native FP4、性能のclaimはない。
- Hot Aisle MI300X x1の結果は、完全なVM/software tuple、single GPU、実行したop/model/shapeだけへ限定する。
  MI300A/MI325X、multi-GPU、bare metalへ自動的に一般化しない。

### Phase 15O model量子化path最適化

- exact `gfx1201`のnative FP8は、同じOCP W8A8/hipBLASLt contractの前段dynamic量子化をwave reduction/native pair
  conversionへ更新した。R9700 Qwen3.5-4B 32/32でprefill `+5.89%`、decode `+10.69%`となったが、BF16よりなお遅く、
  自動provider優先順位は変更しない。exact `gfx1030` emulationも別targetの性能証拠として扱う。
- NVFP4はM=1の従来packed-dequantと、M>1でpacked weight K tileを8 row共有するprefill providerへ分離した。
  M=32 operatorはR9700で59.29〜59.51%、V620で51.21〜56.68%低遅延となり、resident/peak VRAMは不変だった。
  accuracy最大KLD `0.2637523`は既定budget超過のため、当該sLLM製PTQ converter candidateを採用しない。
- Phase 15O期間にはMI300Xが存在せず、新しいexact `gfx942` candidateは有効化していない。R9700/V620の結果を
  CDNA3へ移植せず、Phase 12で検証した既存gfx942 provider scopeも変更しない。

### Phase 15Q Unsloth NVFP4品質attribution

- R9700 exact `gfx1201`とV620 exact `gfx1030`で、Gemma 4 12B-itの同一BF16 source、MLP 144 tensor、BF16
  activation/attention、FP16 KVを固定し、B0/S0/U0/O0を32 prompt・96 logit位置で比較した。両targetとも全dispatch HIP、
  fallbackなし、nonfinite 0、cleanup 0だった。
- U0 Unsloth `imatrix_mse` importはS0 min-maxよりmedian KLDをR9700 `0.3315→0.1619`、V620
  `0.3715→0.1736`へ改善し、top-1一致も`61.46%→79.17%`、`62.50%→76.04%`へ改善した。一方、最大KLDは
  `9.1781`/`7.5777`でbudget `0.05`を大幅に超えた。
- weight MSEだけをbounded searchしたO0は120/144 tensorでS0を改善したが、full-model median KLDは`0.2880`/`0.3433`、
  最大は`14.4025`/`6.4180`だった。activation-aware algorithmの寄与は確認できたが、同一format内で全caseを救済できず、
  S0/U0/O0 converter candidateを採用しない。この結果は提供元QAT/native checkpointのsupport状態ではない。
- この比較はRDNA2/RDNA4のpacked-dequant W4A16だけを証明する。Unsloth公開checkpointのW4A4、attention W8A8、FP8 KV、
  NVIDIA native FP4性能、CDNA3へ一般化しない。

### Phase 16 FP8/NVFP4 KV cache

- exact `gfx1030`/`gfx1201`で、opaque KV stateへのFP8 E4M3FNとpacked NVFP4 append、value/scale plane、
  packed attention direct consumptionを各17 case実行した。独立oracle一致、provider metadata、fallback false、
  cleanup 0を確認し、request全体のFP16/BF16 KV mirrorは作らない。
- KV=8193でallocator granularityを含むcommitted byteはFP16 `18,874,368`に対しFP8 `12,582,912`、
  NVFP4 `10,485,760`だった。短contextではscale planeの最小pageにより理論削減率と一致しない。
- exact `gfx942`はROCm 7.14.0でcompile/linkしただけであり、Phase 12の既存FP16 KV evidenceを低bit KVへ拡張しない。
  通常Qwen loaderはweight metadataからKV encodingを推測せずFP16を維持し、検証済みmodel recipeだけが低bit KVを選べる。

詳細な実装・実機順は[Phase 10](../plans/archive/2026/08/11-20/phase10-fp8-w8a8.md)、
[Phase 11](../plans/archive/2026/08/11-20/phase11-cdna3-port.md)、
[Phase 12](../plans/archive/2026/08/11-20/phase12-mi300x-validation.md)、
[Phase 15O](../plans/archive/2026/08/11-20/phase15o-model-quant-path-optimization.md)、
[Phase 15Q](../plans/archive/2026/08/11-20/phase15q-unsloth-nvfp4-quality-attribution.md)、
[Phase 16](../plans/archive/2026/08/11-20/phase16-kv-cache-fp8-nvfp4.md)を正とする。

## 製品別evidence

2026-08-03時点の事実を次のように記録する。すべて`lifecycle=experimental`であり、evidenceはupstream掲載、local実機G0/G1、未検証範囲を分けて記載する。

| 対象構成 | upstreamの事実 | evidence |
| --- | --- | --- |
| `gfx942`のROCm 7.14.0掲載MI300製品・OS構成 | AMDの現行support資料に掲載 | `[vendor-supported, unverified]` |
| `gfx1200`/`gfx1201`のROCm 7.14.0掲載RDNA 4製品・OS構成 | AMDの現行support資料に掲載 | `[vendor-supported, unverified]` |
| `gfx1030`のROCm 7.14.0掲載Radeon PRO W6800/V620製品・OS構成 | AMDの現行support資料に掲載 | `[vendor-supported, unverified]` |
| canonical Radeon AI PRO R9700 1台、exact `gfx1201` | Ubuntu 24.04.4、kernel `6.17.0-35-generic`、amdgpu `6.16.13`、ROCm 7.14.0でformal model-free G0/G1、Phase 6 A0 HIP VMM PoC、A1比較・KV production probe、A6 Qwen3.5-4B API serviceを実行 | `[project-verified]` |
| canonical Radeon Pro V620 1台、exact `gfx1030` | 同じlocal host tupleでformal model-free G0/G1、Phase 6 A0 HIP VMM PoC、A1比較・KV production probe、A6 API serviceを実行。2台目V620はspareであり必須evidence外 | `[project-verified]` |
| consumer RDNA 2 / RX 6000系 | `gfx1030`の掲載だけでは同じtargetを持つ全consumer SKUの公式対応にならない。[Radeon Linux support matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/native_linux/native_linux_compatibility.html)に完全なSKU/OS構成が掲載された場合だけ`vendor-supported`を付ける | `[unverified]` |
| `gfx1031`–`gfx1036` | LLVMのcode target定義は製品構成のvendor supportではない | `[unverified]` |

Phase 7のlocal draft compileではROCm 7.14.0の`amdclang++`を使い、`gfx1030`〜`gfx1036`、
`gfx1200`、`gfx1201`、`gfx942`の10 exact targetでlink後のCode Object V6 bundleとdevice metadataが
要求targetに一致することを確認した。これはcompile-only draft evidenceであり、この表の
evidence分類や`experimental`のlifecycleを変更しない。

commit `f393d688a051d2b73c8773d8a930a711592609bc`に結び付くformal G0/G1 evidenceの範囲は、canonical exact `gfx1030`/`gfx1201` artifactのidentity/healthと、1、3、17、255、256、257 byteのallocation、copy、diagnostic dispatch、completion、byte-exact copy-backだけである。Code Object V6、wave32、target別ELF flags、artifact metadata、実loader pathを検証したが、このG0/G1 evidence単独ではcapability profile、resource gate、semantic数値kernel、model、性能を証明しない。後続A0/A1 evidenceは以下に別scopeとして追加し、いずれもtarget全体や別SKUへ一般化しない。完全なsoftware tupleと実行結果は[software compatibility](software.md)に記録する。

Phase 6 A0では同じcanonical 2 targetについてHIP VMM capability、minimum/recommended granularity、
VA reserve、physical create/map/access、contiguous-pointer kernel、unmap/remap、event lifetime、cleanupを
model-freeに追加検証した。両targetのminimumは4 KiB、recommendedは2 MiBで、Qwen3.5-4B相当16 regionの
2 MiB page activation p95はV620 582.841 us、R9700 496.488 usだった。このevidenceはVMM primitiveの
`project-verified`であり、vAttention production backend、full model、他RDNA2/RDNA4 SKU、別ROCm/kernel、
Paged Attentionとの方式選択を証明しない。

Phase 6 A1では同じ2 targetでQ heads 16、KV heads 4、head dimension 256、Q length 1/37、
KV length 255/256/257/1023/1024/1025のcontiguous、vAttention、paged accessorを同じ
FA2-style online-softmax proxyで実行した。NumPy oracle、mode間数値一致、fallbackなし、health、cleanupを
PASSし、actual public runtimeのvirtual-contiguous KVも1023/1024/1025 tokenで全要素oracle、
2/2/4 MiB per-plane commitment、未map readback拒否を両targetでPASSした。このscopeについて
`project-verified`とし、初期方式はvAttention型を選択した。

これはupstream FlashAttention-2/CKの性能、FlashAttention-3/4のAMD動作、full model、長時間安定性、
他SKU/tuple、Paged Attention production backendを証明しない。実測値、identity、再検討条件は
[KV memory decision](../architecture/kv-memory.md)を正とする。

### 2026-08-14 Phase 8 BF16 optimized-path evidence

同じlocal tupleのcanonical V620 `gfx1030` / R9700 `gfx1201`で、BF16 activation/weight、FP32
accumulation、BF16 RNE outputのMatmul registryと、vAttention上のFA2-style causal attentionを実行した。
Matmulはfrozen numerical manifestの5形状を含む17 caseでspecial classification、exact target、provider ID、
fallbackなし、cleanup 0をPASSした。M=1,K=2560,N=9216はV620でcustom workgroup reduction、R9700で
hipBLAS GEMMExを選び、weight copy/workspaceは0である。hipBLASの採用は`gfx1201`のM=1,K/N>=1024に
限定し、V620や別RDNA4 SKUへ一般化しない。

causal attentionはhead dim 256の協調dot reductionとonline softmaxを使い、prefill M=1/3/17/37、
committed KV 1023/1024/1025、NaN/+Inf classificationを含む16 caseで独立oracleをPASSした。
virtual-contiguous FP16 K/V、opaque owner、GQA mappingはPhase 6契約と同じである。これはproduction
FA2-style scopeの`project-verified`であり、upstream FA2/CKの同等性能、Paged Attention、FA3/4、
他targetを証明しない。RDNA4向けFA3-likeはPhase 8完了をblockしない将来taskである。

Phase 6 A6ではQwen3.5-4B lock
`sha256:f143d7b504170d071c77818105f7a07dc0297c6bea0c61a5404b071fed0c1fae`を使い、canonical
V620/R9700の各UUIDだけを`ROCR_VISIBLE_DEVICES`で可視化して論理device 0へ接続した。両targetで
non-stream、SSE、stop、HIP dispatch後disconnect、1023/1024/1025 logical capacity、2 MiB physical page、
32 MiB committed K/V、request-local KV/linear・VRAM・GPU process cleanupをPASSした。target別binary SHA-256は
`gfx1030=dd07bca6c1ca023365bc8800142302929ee50495993e431843aa35528b81723c`、
`gfx1201=029fdf71f5899200915f1f8a5161316c6f9832f85dbb3ea9a7ddc188c677067b`である。

このA6 scopeは単一GPU・単一model-resident serverだけである。HIP current deviceはthread-localなので、
mixed-GPU hostで複数GPUを可視化したままglobal physical indexを渡す構成は対応外とし、stable UUIDによる
単独可視化と論理device 0を必須運用条件とする。他SKU、別tuple、multi-GPU、長時間安定性へ一般化しない。

### 2026-08-14 Phase 9 engine-structure evidence

同じlocal tupleとcanonical V620 `gfx1030` / R9700 `gfx1201`で、HIP Graph capture/replay、BF16
M=1 MMVF v3、Qwen3.5 GDN private state layout、same-stream completion segment、target別prefill providerを
実行した。HIP Graph PoCはsLLM kernel 1 nodeとhipBLAS混在2 nodeでpointer/scalar更新、独立oracle、cleanupを
両targetでPASSした。productionはrequestごとのgraph instantiateを行わず、KV appendを明示境界とする
segment pathを採用したため、PoC結果をfull production graph対応へ一般化しない。

Matmul 17 caseは両targetでexact target、numerical/classification oracle、fallbackなし、cleanup 0をPASSした。
M=1,K=2560,N=9216はV620 259.282 us、R9700 75.002 usで、M=1は両targetのMMVF v3を選択する。
M>1はV620のtiled16を維持し、R9700だけcontext-lifetime hipBLAS GEMMExを選択する。GDN stateはV620だけ
wave-coalesced transposed layout、R9700はthread-contiguous rowであり、別RDNA2/RDNA4 targetへ推論しない。

4B short-odd 3 warmup + 10 measured中央値はV620がTTFT 0.306 s、prefill 56.91、decode 29.69 tok/s、
E2E 0.855 s、R9700が0.051 s、377.46、37.20 tok/s、0.490 sだった。HIP-only、fallbackなし、resident/peak
VRAMは8,411,592,192/8,540,569,292 bytes、cleanup 0である。32/32、2B V620、9B R9700、R9700の
OpenAI non-stream/SSEもPASSした。scopeはQwen3.5 BF16 single requestと明示したshapeだけであり、
multi-request、production全体のgraph replay、別SKU/tuple、長時間安定性を証明しない。

残差profileはmemory-bound M=1 matvecを支配要因とし、full attentionは支配的でなかった。RDNA4 FA3-likeは
引き続き将来のtarget-specific比較であり、Phase 9 evidenceへ含めない。詳細値とdigestは
`ci/matrix/phase9-profile-summary-v1.json`を正とする。

### 2026-08-16 Phase 16F mixed NVFP4 evidence

同じlocal tupleのV620 `gfx1030`とR9700 `gfx1201`で、dynamic block-16 NVFP4 W4A4 12境界case、static FP8 KV
17 case、Unsloth Gemma 4 12B mixed full graph 8 transitionをPASSした。full graphはresident 9,201,189,600 byte、
peak accounted 9,221,491,952 byte、fallbackなし、cleanup 0で、両targetの生成token列は`[532; 8]`だった。
R9700は既存単一GPU contractに従い`HIP_VISIBLE_DEVICES=2`で単独可視化し、論理device 0へ接続した。

これはpacked-dequant/packed-direct software providerのproject evidenceであり、RDNA native FP4 instruction、別SKU、別ROCm tuple、
multi-GPU、same-artifact NVIDIA reference correctnessを証明しない。reference未実行のmodel evidenceは`experimental`とする。

## 将来AMD候補

初期範囲外であっても将来対応の意図があるものは`unsupported`ではなく`lifecycle=planned, evidence=[unverified]`とする。

| 表示分類 | target例 | lifecycle | evidence |
| --- | --- | --- | --- |
| RDNA 3 | `gfx1100`–`gfx1103` | `planned` | `unverified` |
| RDNA 3.5 | `gfx1150`系 | `planned` | `unverified` |
| MI50 | `gfx906` | `planned` | `unverified` |
| CDNA 1 | `gfx908` | `planned` | `unverified` |
| CDNA 2 | `gfx90a` | `planned` | `unverified` |
| CDNA 4 | `gfx950` | `planned` | `unverified` |
| CDNA 5 | target未確定 | `planned` | `unverified` |

未知のAMD targetは未分類であり、自動的に`unsupported`とはしない。
