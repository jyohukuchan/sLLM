# AMD GPU互換性方針

> 最終更新: 2026-08-24
>
> この文書はAMD向けの識別規則と初期候補を記録する。現時点の初期targetはすべて`lifecycle=experimental`である。計画targetのevidenceは`unverified`、canonical local実機のformal model-free G0/G1とPhase 6 A0 HIP VMM PoCは検証した限定範囲だけ`project-verified`とする。Phase 49のGQA P32はexact `gfx1030`限定、Phase 50のResidual/GDN/MLP/P32はexact `gfx1201`の狭いscope限定であり、target全体やSKU全体の昇格ではない。

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

### 2026-08-16 Phase 17 Qwen3.5 MTP/vision evidence

同じlocal tupleのV620 `gfx1030`とR9700 `gfx1201`で、fixed Qwen3.5-4BのMTP 15 tensorとvision 297 tensorを
real-weight実行した。MTPは両targetでdraft/verify token、accepted prefix、deterministic replay、HIP-only、fallbackなし、cleanup 0を
PASSしたが、逐次target verifyのため性能採用せず通常providerはtarget-onlyとした。visionは256 patch/64 visual token、projector、
3-axis mRoPE text prefill、1 decodeを両targetでPASSした。targetごとのprojected digestはtarget内で再現し、異なるHIP数学provider間の
bit一致は要求しない。別SKU、別画像geometry、video、low-bit visionへ一般化しない。

### 2026-08-16 Phase 18 exact MTP evidence

同じlocal tupleのcanonical V620 `gfx1030`とR9700 `gfx1201`で、fixed Qwen3.5-4Bのtarget block M=`2/3/4/7/8`を
BF16+FP16 KVおよびFP8 W8A8+static FP8 KVで実行した。全幅のtoken/hidden、M=8のraw target logitsとaccepted-prefix K/V
payloadは逐次M=1とbit/byte exactで、全dispatch HIP、fallbackなし、cleanup 0だった。R9700 BF16 width 1の3 warmup + 10 measuredは
speedup中央値`1.0355x`、MAD`0.0028`、p10/p90 `1.0242/1.0448`だった。V620 screening中央値`0.9990x`はnoise内なので
正確なprovider実装だけを保持し通常auto-selectionは行わない。別tuple、model、sampling、vision、長時間運転へ一般化しない。

### 2026-08-16 Phase 19 Qwen3.5 MoE evidence

同じlocal tupleのcanonical R9700 `gfx1201`（UUID `GPU-a8e9ddefa2d60f55`）とV620 `gfx1030`
（UUID `GPU-08b2ddcbd6e6b36c`）で、fixed `amd/Qwen3.5-35B-A3B-MXFP4` text-only artifactを実行した。
router境界matrixとactual-weight expert oracle（layer 0/19/39、M=1/3/7、expert 0–7/124–131/248–255）は
最大誤差`1.86265e-9`、active pair 8/24/56、fallback 0をPASSした。full modelはprefill/decodeのSparseMoeを
40/40回、active pairを960/320とexactに監査し、両targetで同じprefill/decode tokenとreplayを得た。

resident currentは22,009,574,016 byte、request stateは129,474,560 byte、workspaceは17,982,024 byte、
high-waterは22,230,758,892 byteだった。2 warmup + 11 measuredのprefill/decode中央値はR9700が
216.258/204.198 ms、V620が537.832/370.711 msである。通常CLI/API、SSE、cancel/recovery、seeded sampling、shutdownを
HIP-only、fallbackなし、cleanup 0でPASSした。別SKU/tuple、MoE vision/MTP、multi-GPU、batchingへ一般化しない。

### 2026-08-17 Phase X llama.cpp Qwen3.8 quantized-KV evidence

canonical R9700 `gfx1201`（UUID `GPU-a8e9ddefa2d60f55`）とspare V620 `gfx1030`
（UUID `GPU-08b2ddcbd6e6b36c`）で、fixed Qwen3.8-27B Q5_K_XL、Q5_1 model/draft KV、MTP幅3、
context 262,144のllama.cpp HIP/Vulkan controlを実行した。HIP baseline低下の根因はGDNではなく、
`GGML_CUDA_FA_ALL_QUANTS=OFF`でQ5_1 K/VがFlash Attentionから外れたことだった。`ON`のfresh HIP buildは
9,435-token promptのprefill/decode中央値がV620 340.80/33.42、R9700 779.06/41.93 tok/sとなった。

Qwenのhead dimension 256、GQA比6、KV長113/512/1024、query batch 1/3/17を覆うQ5_1 Flash-Attention testは、
CPU numerical oracleに対して両target各18/18 PASSした。peak VRAMは約30.1/30.2 GBで、CPU/backend fallbackと
GTT spillはない。このevidenceは外部llama.cpp local-subagent runtimeと固定tupleだけに限定し、sLLMのFP16 KV/GDN、
Vulkan backend support、別model/SKUまたは一般的なquantized-KV supportへ一般化しない。詳細は
[Phase X bounded summary](../../ci/matrix/phase-x-qwen38-amd-summary-v1.json)を正とする。

非運用のV620×2 tensor-split controlではactual context 1,048,576を確保し、同じ9,435-token code promptの
1 warmup + 3 measured中央値がprefill/decode 416.80/47.90 tok/sだった。request中の合計observed peak VRAMは
66,560,937,984/68,685,922,304 byte（61.99 GiB、96.91%）、headroomは2,124,984,320 byte、GTTは
40,599,552 byteである。2基間は2-hop PCIeで、internal AllReduceは使用できずmeta-backend butterflyへ移行し、
token samplerはCPUへfallbackした。全model layerとtarget/draft contextはGPU residentだが、1M-token実入力、
GPU-only end-to-end、strict scalingまたは長時間安定性の証拠ではない。この計測後のユーザー決定で同じV620×2 tensor shapeを
491,520 context/slot、983,040 totalへ縮小し、parallel 2、non-unified KVの通常local-subagent構成へ昇格した。
現行構成はidle時約2.48 GB/GPUのheadroomを持ち、2つの同時taskを別slotで完了し、3つ目をqueueせず拒否することを確認した。
[TP2 1M bounded summary](../../ci/matrix/phase-x-qwen38-v620-tp2-1m-summary-v1.json)は昇格前の計測値、
[運用正本](../development/local-qwen-subagent.md)は現行値を管理する。

同日の2要求follow-upでは、独立V620 server 2基が368,640 context/要求で45.58/47.01秒、V620×2 tensor splitが
524,288 context/slotで59.14/60.78秒だった。V620×2 layer splitは1M totalでMTP compute-buffer OOMとなり、
917,504 totalへ縮小しても67.31/69.56秒だった。R9700+V620×2はmulti-target HIP buildで動作し、layer split
`5,2,2`が524,288 context/slot、45.82/47.90秒、peak VRAM 30.97/16.14/21.52 GBでbounded比較の最良
single-process profileとなった。ただしR9700をsLLM開発から占有するため非運用とする。3基tensor splitは
63.45/65.08秒で棄却した。この比較後に上記V620×2 tensorの縮小構成だけを通常運用へ昇格した。詳細は
[multi-GPU selection summary](../../ci/matrix/phase-x-qwen38-multi-gpu-selection-v1.json)を参照する。

Phase 20では同じcanonical V620 `gfx1030`とR9700 `gfx1201`で、単一GGUFからQwen3.5-4B BF16、
Gemma 4-12B mixed NVFP4、Qwen3.5-35B-A3B MXFP4をfull-model実行し、source importerと同じtop-1、HIP-only、
fallbackなし、cleanup 0を確認した。R9700ではQwen FP8 GGUFも同条件をPASSした。MoEのOpenAI serverはR9700で
model list、chat completion、graceful shutdownをPASSした。max-rank-4再生成後もMoEを両target、FP8をR9700で再実行し、
Qwen BF16 visionの233-token prefillを両target、GGUF MTPとQwen OpenAI serverをR9700でPASSした。Qwen BF16の
3 warmup + 10 measured固定laneはR9700/V620でload 10.654/10.331 s、median TTFT 46.653/184.143 ms、
median TPOT 26.689/29.685 msだった。この証拠は固定artifact、single request、実行済みtargetに限定する。

### 2026-08-19 Phase 30 RDNA4 attention/KV evidence

canonical R9700 UUID `GPU-a8e9ddefa2d60f55`のexact `gfx1201`で、causal attentionのE4M3FN readを
`v_cvt_f32_fp8`へ置き換え、256-thread LDS reductionをwave32 shuffleと8 wave partialへ変更した。
`query_count=1`と`query_count>=32`だけがtarget-scoped providerへrouteし、2〜31およびcanonical V620 UUID
`GPU-76a08c022586fed6`のexact `gfx1030`はbaselineを維持する。gfx1201/gfx1030 × FP16/FP8の各17 caseは
全出力一致、fallback 0、cleanup failure 0だった。gfx1201 native decodeは全256 E4M3FN code（NaN 2 codeを含む）を
software contractへbit exact照合し、code objectで`v_cvt_f32_fp8`とwave shuffleを確認した。gfx1030 code objectへ
native FP8命令はない。

Qwen3.5-4B BF16、4108 input、output 32の3 independent process中央値はgfx1201 baseline比でTTFT 9.60%、
prefill 9.72%、E2E 9.16%、decode throughput 7.86%改善した。10000+ inputはfull-prefill workspace preflightが
53,758,880,592 byteを要求し、available 34,135,343,104 byteでは起動しなかったため性能PASSへ数えていない。
native append encodeはchunk 256で68.69%悪化して棄却し、software encoderを維持した。prefill matrix/FlashAttention
providerも本Phaseでは採用していない。詳細は[Phase 30 summary](../../ci/matrix/phase30-rdna4-attention-kv-summary-v1.json)を正とする。

### 2026-08-19 Phase 31 chunked prefill・low-bit KV evidence

Phase 30と同じQwen3.5-4B BF16 GGUF/lock、ROCm 7.14.0、canonical exact gfx1201/gfx1030をtarget別release buildで
実行した。completion-boundary liveness arenaにより10,001-token workspaceは個別allocation合計39,950,821,120 byteから
high-water 5,278,049,280 byte相当へ縮小し、従来53.76 GB requiredで拒否された10k+ full-modelを両targetで完走した。
gfx1201の16,385 tokenは16,384+1の2 chunkとなり、workspace high-water 8,646,688,768 byte、HIP-only、fallbackなし、
cleanup 0だった。gfx1030/gfx1201の10,001-token dynamic FP8 KVも1 decode stepを含めて同条件をPASSし、gfx1201では
16,385-token dynamic FP8と10,001-token static FP8もPASSした。NVFP4はgfx1201の513-token spotに限定する。

この証拠はchunk/arenaのmemory feasibilityとlow-bit routingの限定証拠であり、single-run timingを安定した速度比較、
low-bit品質、別model/SKU/tupleへ一般化しない。KV providerはvirtual-contiguousのままで、Paged Attention、native append encode、
MTP/multimodal/MoE low-bit、default FP8化は含まない。詳細は
[Phase 31 summary](../../ci/matrix/phase31-chunked-prefill-summary-v1.json)を正とする。

### 2026-08-19 Phase 32 native FP8 append evidence

canonical R9700 UUID `GPU-a8e9ddefa2d60f55`のexact `gfx1201`で、dynamic/static FP8 KV appendの
最終E4M3FN encodeをnative scalar conversionへlowerした。production code objectは`v_cvt_pk_fp8_f32`を2命令含み、
canonical V620 UUID `GPU-76a08c022586fed6`のexact `gfx1030` code objectには同命令がない。kernel symbol、256-thread
workgroup、grid、scale、store、KV layout/publication、public ABIは共通のままである。

全BF16 codeをK/Vで一巡したprototypeと19 token境界はpayload/scale mismatch 0、production attention oracleは
gfx1201/gfx1030 × dynamic/static FP8の68/68 caseをPASSした。gfx1201の10,001-token production append familyは
4,520,428 nsから2,191,564 nsへ51.52%短縮したが、full-model寄与は通常のtiming noise以下なのでuser-visible speedupを
claimしない。gfx1201 10,001/16,385、gfx1030 10,001 inputはtoken `[1228, 1228]`、HIP-only、fallbackなし、cleanup 0だった。
native packed/128-thread候補、gfx1030 native化、default FP8化は採用していない。詳細は
[Phase 32 summary](../../ci/matrix/phase32-native-fp8-append-summary-v1.json)を正とする。

### 2026-08-20 Phase 33 Full Attention evidence

canonical R9700 exact `gfx1201`とV620 exact `gfx1030`へ、scratch-freeな共通Full Attention providerを限定採用した。
C1は`M=1`/KV>=1,024を8 waveの連続KV区間へ分割し、QK依存深さ8→12のN2を明記したうえでユーザー承認を得た。
C2は`M>=64`/GQA4/head dim 256で4 query headのK/V decodeを共有する。両者ともglobal scratchと追加dispatchは0で、
scope外はPhase 30/B0へ事前routeする。FP16/dynamic FP8/static FP8/NVFP4の232/232 oracle、representative full-model、
dynamic FP8 API lifecycle、wrong-target拒否を最終binaryでPASSし、fallback/cleanupは0だった。C3 matrix innerは採用C2の
4-row tileが16×16×16 WMMAへ合わないため棄却した。詳細は
[Phase 33 summary](../../ci/matrix/phase33-full-attention-summary-v1.json)を正とする。

### 2026-08-20 Phase 34 V620 long-prefill BF16 matmul evidence

canonical V620 UUID `GPU-76a08c022586fed6`のexact gfx1030でcurrent tiled16とexisting hipBLASを同一buffer/stream比較した。
10,001行の248 projection加重値は62.526秒から11.081秒へ82.28%、Qwen3.5-4B FP16 KV full modelは
89.249秒から34.684秒へ61.14%短縮した。主要5 shapeを`M>=128`、K/V shapeを`M>=1024`でrouteし、N=32と
scope外shapeは旧providerへ残す。canonical R9700 UUID `GPU-a8e9ddefa2d60f55`の10,001-token controlは75.316秒、
既存hipBLAS route不変で、両targetとも同token、HIP-only、fallbackなし、cleanup 0だった。

selected solutionはGSU1、global atomic combineなしで、stress oracleのbound違反は0だった。gfx942はcompile-only、gfx1201 binaryの
V620 loadはexact target mismatchで拒否された。この証拠を別V620/RDNA2 SKU、別model shape、ROCm version、N=32やunknown shapeへ
一般化しない。[Phase 34 summary](../../ci/matrix/phase34-v620-prefill-matmul-summary-v1.json)を正とする。

### 2026-08-20 Phase 35 long-context Full Attention/GDN evidence

canonical V620 exact `gfx1030`とR9700 exact `gfx1201`へ同じupper sourceの二providerを限定採用した。Full Attentionは
GQA4/head dim 256の`M>=128`をQ_TILE=4へ送り、4 query rowがK/V decodeを共有する。GDNはQ/K 16、value 32、
head dim 128のtoken count 128以上を1,024-workgroup column-state pipelineへ送る。短M/decode、別shape/targetは既存routeを
維持し、runtime error後fallback、KV/state layout migration、global attention scratch、weight repackは追加していない。

両target・4 KV encoding・各29 caseのFull Attention 232/232と、両target・token 1/3/17/127/128/129のGDN 12/12を
独立oracleでPASSした。final 10,001 input / 2 outputのcombined E2EはV620 34.861秒から22.683秒へ34.93%、
R9700 75.349秒から65.214秒へ13.45%短縮し、token `[2064,5686]`、HIP-only、fallback false、cleanup 0、arena不変だった。
V620 profileではFull Attentionが10.820秒から4.110秒、GDN familyが約7.672秒から0.618秒となった。これはexact tupleと
Qwen3.5-4B固定shapeの証拠であり、別SKU/model/ROCmやFull Attention peer parityへ一般化しない。
[Phase 35 summary](../../ci/matrix/phase35-attention-gdn-summary-v1.json)を正とする。

### 2026-08-21 Phase36 Session A: MI300X VF exact tuple

Hot Aisle MI300X VF x1 の `gfx942:sramecc+:xnack-`、wave64、304 CU、HBM `205,822,885,888` bytes、NPS1/SPX、
VMM `true` を、Ubuntu 24.04.4 / kernel `6.8.0-124` / amdgpu `6.16.13` と ROCm 7.14.0、HIP `7.14.60850`、
LLVM 23 の一つの compatibility tuple として検証した。release artifacts はlogical `gfx942`に対して唯一のdevice bundle
`gfx942:sramecc+:xnack-`、Code Object V6、ELF flags `0xE4C`（SRAM ECC on / XNACK off）、全kernel wave64であり、
generic code object や別targetを含めない。wrong-targetはdispatch前に拒否し、最終artifactのoperator matrixは99/99 PASSである。

Qwen3.5-4B BF16/FP8 GGUF は HIP-only、fallback `0`、cleanup `0` で固定短生成を完了した。gfx942 FP8 は OCP E4M3FN の
storage bytes/scales を native FNUZ resident 表現へ実変換する（label-only/raw reinterpret ではない）。Hello の BF16 token は
`[11,353,2688,4313,310]`、FP8 は `[11,353,1044,4313,310]` で、cross-provider N1 差を記録する。Unicode と stop も同一
target/provider contract で PASS した。Phase29 の GDN wave32 scope leak は修正され、wave64 では sequential norm を維持する。
BF16/FP8のsecond resident requestとmodel reuse、drop後0を確認した。post-runはprocess `0`、全sysfs RAS block CE/UE `0`、
VRAM baseline、provider `/opt/rocm` link復元、VM外raw退避を確認した。

これは Session A のみの `project-verified` evidence であり、Sessions B-D、9B、low-bit KV/long context、MTP、vision、service、
performance、multi-GPU、別SKU/VM/bare-metal への対応宣言ではない。lifecycle は `experimental` のままとする。

### 2026-08-21 Phase36 Sessions B/C: MI300X current runtime

Session Aと同じexact MI300X VF / `gfx942:sramecc+:xnack-` / wave64 / ROCm 7.14 tupleで、4 KV encodingの
Full Attention 116/116、FP16 KV state 19/19、low-bit独立oracleをPASSした。Qwen3.5-4B BF16 targetの
FP16/dynamic FP8 KVを
autoおよび512/2,048/4,096/8,192/16,384 token chunkで実行し、12/12 rowがexact 10,001 input / 2 output、
token `[23066,23066]`、HIP-only、fallbackなし、cleanup 0だった。request stateはFP16 `379,289,600`、dynamic FP8
`217,961,216` bytes、arena high-waterはauto/16K `5,278,049,280`、512 `270,209,024` bytesである。
`contiguous-resident`の物理HBM/GTTを測定し、終了後baseline復帰を確認したが、VMM provider自体は変更していない。

MTPはBF16+FP16 KVのoff/width 2/3/4/7/8と、FP8 target+dynamic FP8 target KVのoff/width 3をPASSした。
forced MTPのexact gfx942、width 1〜8、bounded state slack、quantized-plan admissionを追加し、visible token、proposalの
accepted/rejected accounting、cleanupをtarget-onlyへ照合した。FP8 target時もMTP side weights/KVはBF16/FP16である。
PNG/JPEG/WebP各64 visual token、lazy resident reuseとshutdown、OpenAI profile v1のnon-stream/SSE/reasoning/stop/seed/
cancel-recovery/two-concurrent/graceful shutdownをPASSした。Session D、9B、performance、llama.cpp/profile、別SKU/ROCm
tupleを含まず、lifecycleは`experimental`のままとする。詳細は
[Session B summary](../../ci/matrix/phase36-mi300x-session-b-summary-v1.json)と
[Session C summary](../../ci/matrix/phase36-mi300x-session-c-summary-v1.json)を正とする。

### 2026-08-21 Phase36 Session D: MI300X performance/profile evidence

同じexact MI300X VF / `gfx942:sramecc+:xnack-` / wave64 / ROCm 7.14 tupleで、Qwen3.5-4B BF16/FNUZ FP8の
各5ケースを3 warmup＋10 measuredでPASSした。10,001 input / 2 output E2E中央値はBF16 `22.556130816`秒、
FP8 `22.556528472`秒で、exact HIP-only、fallbackなし、終了後HBM/GTT baseline復帰を確認した。fixed llama.cpp
`b10453` / `3cb7ffb1`の同条件E1比較は`0.8512540725`秒、sLLM比`26.4975`だった。BF16 10,001/2の
rocprofv3 device shareはGDN `73.95%`、Full Attention `25.12%`、projection `0.70%`、other `0.23%`である。
この結果は単一VFと当該model/shapeの性能evidenceであり、別CDNA SKU、multi-GPU、一般性能保証へ一般化しない。
Sessions A〜DをPhase 36の完了範囲としてcloseし、VMはユーザーが削除した。lifecycleは`experimental`のままとし、
[Session D summary](../../ci/matrix/phase36-mi300x-session-d-summary-v1.json)を正とする。

### 2026-08-21 R9700 sLLM / fixed llama.cpp direct E2E

canonical R9700 UUID `GPU-a8e9ddefa2d60f55`、BDF `0000:07:00.0`、exact `gfx1201`で、Qwen3.5-4B BF16、
FP16 KV、`23066`×10,001 input / 2 output、greedy、3+10を実行した。current-source sLLM / fixed llama.cpp
`b10453`のE2E中央値は`3.936429665/2.063845785`秒、比`1.90733x`である。生成tokenは全試行
`[23066,23066]`、sLLMはfallback/cleanup 0、llama.cppは33/33 layer full GPU offload、終了後process 0だった。
GGUF identityは異なるためE1 system-equivalentに限定する。Phase 35のR9700 messages rowは入力・生成が異なり、
この比率へ混ぜない。[R9700 summary](../../ci/matrix/r9700-sllm-llama-e2e-v1.json)を正とする。

### 2026-08-21 Phase40 token selector・grammar scope（verification中）

Phase 40のhost/API実装は、ordered sampler chain、bounded GBNF/JSON Schema、raw-byte/token-trie/partial UTF-8 state、post-mask
logprobs、`n=1..=8` choice stateを提供する。HIPは既存Argmax ABIと分離したadditive `TokenSelect` contractを実装し、Qwen/Gemmaの
terminal projectionと同じqueueでF32 additive/U8 valid-maskを使い、completion後に固定16-byte selected recordだけをreadbackする。
これはPhase40全体の実機対応宣言ではなく、selector contract matrix取得済み・sampled-generation integration継続中の限定statusである。

| exact target | Phase40 lifecycle | Phase40 evidence | status |
| --- | --- | --- | --- |
| V620 `gfx1030` | `experimental` | `project-verified`（selector scope） | vocab `1,3,17,255,256,257,248320`×counter `0,1`、CPU token/logprob、fallback 0、selected-only D2H 16 bytes |
| R9700 `gfx1201` | `experimental` | `project-verified`（selector scope） | V620と同一matrix、CPU token/logprob tolerance `.005`、fallback 0、selected-only D2H 16 bytes |
| MI300X `gfx942:sramecc+:xnack-` | `experimental` | `unverified` | wave64 feature-pinned compile-only PASS。real correctness/performanceはVM再確保後へdeferred |

V620/R9700 selector matrixはodd vocabulary・mask/bias、NaN/Inf/all-mask、fixed seed、CPU oracle（logprob tolerance `.005`）、fallback 0、
selected record D2H 16 bytes、full-vocabulary D2H 0を記録済みである。GPU unavailable、timeout、crash、zero selectionはPASSとしない。
Qwen/Gemma sampled-generationの最終統合runとPhase40全体のrelease reviewは別途継続する。既存R9700 10,001/2 E1 E2EはPhase40 selector
evidenceとは別である。Phase40ではllama.cpp sourceの直接reuseはなく、provenance lockは変更しない。詳細は
[archive plan](../plans/archive/2026/08/21-31/phase40-token-selection-grammar-structured-generation.md)と
[history](../history/2026/08/21-31/phase40-token-selection-grammar-structured-generation.md)および
[tracked GPU summary](../../ci/matrix/phase40-token-selector-gpu-summary-v1.json)を参照する。

### 2026-08-22 Phase41 state fork/COW/image

canonical V620 `GPU-76a08c022586fed6` / exact `gfx1030`とR9700 `GPU-a8e9ddefa2d60f55` / exact `gfx1201`で、
opaque stateのreal-GPU matrixを実行した。FP16 VMM fork/COWは63/64/65/127/128/129 token境界、source/child K/V、
child append後のsource不変をbyte exactで確認した。dynamic/static FP8とNVFP4は全2/4/6 plane、linear attentionはactive slotと
scratchを含む5 planeをimport/fork/export oracleへ照合した。各binaryは14 bundleが指定targetだけを含み、fallback、cleanup failure、
終了後GPU process、uncorrectable ECCはいずれも0だった。

exact `gfx942:sramecc+:xnack-`はROCm 7.14/LLVM 23、wave64でcompile/linkだけをPASSした。MI300X VMは削除済みなので
Phase 41 state/context/checkpointのreal gfx942 evidenceは追加せず、既存Phase 36の別scopeへも遡及しない。lifecycleは全targetで
`experimental`のままで、証跡範囲は[Phase41 GPU summary](../../ci/matrix/phase41-state-gpu-summary-v1.json)を正とする。

### 2026-08-24 Phase49 exact `gfx1030`限定性能scope

Phase 49はcanonical V620 exact `gfx1030`（UUID `GPU-76a08c022586fed6`、BDF `0000:03:00.0`）だけへ性能候補を開き、GQA4 decodeのP32 partitionを`M=1`、
head dimension 256、FP16 KV、KV長4,096以上へ限定して既定採用した。long-prefill v2とHIP Graphは採用せず、
このtargetのselector、閾値、solution ID、wave32 binaryをexact `gfx1201`または`gfx942`へコピーしない。
最終通常5行の正しさ・HIP-only・fallback・cleanupを確認したが、全7行の同等達成は主張しない。この履歴の
`project-verified`はV620の上記scopeだけに付与し、RDNA2全SKUや`gfx1030`全shapeへ一般化しない。詳細は
[数値変更台帳](numerical-output-changes.md)と[Phase 49以降ロードマップ](../history/2026/08/21-31/phase37-plus-mi300x-and-llama-gap-roadmap.md)を参照する。

### 2026-08-24 Phase50 exact `gfx1201`狭い実機scope

canonical Radeon AI PRO R9700のUUID `GPU-a8e9ddefa2d60f55`、BDF `0000:07:00.0`、exact `gfx1201`だけを
Phase 50の性能targetとした。Residual RMSNorm、GDN projection、MLP gate/up/SiLU、GQA P32のA/B比較を、
Qwen3.5-4B BF16、FP16 KV、`M=1`、単一active requestのtarget専用selectorで確認した。最終通常行と長行の
計測・未達判定はこのR9700 scopeへ限定し、100,000-token prefillはOOMで完走せず`project-verified`成功scopeへ含めない。
20,000-token decodeの完走結果は[Phase 50履歴](../history/2026/08/21-31/phase50-r9700-port-and-mi300x-handoff.md)とsummaryへ固定し、
この方針文書では数値を重複しない。詳細なcandidate分類と取得済み行は[数値変更台帳](numerical-output-changes.md)を正とする。

| target / scope | lifecycle | evidence | status |
| --- | --- | --- | --- |
| R9700 exact `gfx1201` Phase 50 Residual/GDN/MLP/P32 A/B、通常・長行 | `experimental` | `project-verified`（上記scopeのみ。100k OOM未達は除外） | HIP-only、fallbackなし、target selector、cleanup/資源復帰を確認。別shape・別model・別SKU・別tupleへ一般化しない |
| MI300X logical `gfx942` / feature付きdevice target `gfx942:sramecc+:xnack-` Phase 50 handoff | `experimental` | `unverified`（compile/host scope） | `sramecc=on`、`xnack=off`、Code Object V6、wave64のcompile/linkとhost selector非選択のみ。MI300X runtime PASSはPhase 51実機待ち |

Phase 50のR9700固定tupleはUbuntu 24.04.4、kernel `6.17.0-35-generic`、amdgpu `6.16.13`、ROCm 7.14.0、
HIP `7.14.60850`、LLVM 23、Code Object V6、wave32である。exact target専用artifactとloader closureを使い、
generic/multi-arch binaryを性能evidenceへ混ぜない。MI300X側は論理target `gfx942`、feature付きdevice/codegen target
`gfx942:sramecc+:xnack-`、Code Object V6、wave64、`xnack=off`、`sramecc=on`をcompile/linkへ固定する。production Cargoの
`CMAKE_HIP_ARCHITECTURES`はlogical `gfx942`を使い、feature suffix付きtargetはdirect CMake probeだけで扱う。gfx1201 providerがhost selectorで非選択になることだけを検証する。Phase 50のcompile/host結果は既存Hot Aisle
MI300X runtime evidenceを拡張せず、Phase 51でfresh preflightと実機7行を行うまで実行対応とは扱わない。
すべてのlifecycleは`experimental`のままとし、RDNA4/CDNA3の広いtarget、SKU、OS、driver、ROCm tupleへ昇格しない。

### 2026-08-24 Phase52 exact `gfx1201`長capacity resident KV

Phase 50後の自動2K再実行はHBM総容量を大きく下回るlayer 23 VMM physical commitで失敗した。Phase 52では
exact `gfx1201`かつlogical capacity 65,536以上だけを`contiguous-resident`へ固定し、65,535以下、unknown target、
他targetのcapability-selected policyを変更しなかった。VMM grow/COWは途中失敗をappend前へ戻すtransactional rollbackへ変更した。

canonical R9700で`100,000/2`を1 warmup＋3 measured、`10,001/2`を3 warmup＋10 measured実行し、それぞれ4/4、
13/13 PASSした。100kは自動chunk 2,048、8 KV layer、K/V 4 GiB、E2E中央値`325.593963905`秒、HBM peak
`15,388,794,880` bytesである。両行で生成`[23066,23066]`、HIP-only、fallback/cleanup 0、process消滅、baseline復帰を確認した。
この固定scopeだけを`project-verified`へ追加し、lifecycleは`experimental`を維持する。詳細は
[Phase 52 summary](../../ci/matrix/phase52-r9700-kv-commit-summary-v1.json)を正とする。

| target / scope | lifecycle | evidence | status |
| --- | --- | --- | --- |
| R9700 exact `gfx1201` Phase 52 Qwen3.5-4B BF16 `100,000/2` | `experimental` | `project-verified`（固定長context scope） | resident KV、自動2K、4/4 PASS、HIP-only、fallback/cleanup 0、資源復帰 |

### 2026-08-27 Phase53 descriptor v1判定履歴とv2 follow-up

R9700 exact `gfx1201`はdescriptor v1の`kv-fp8-e4-block16`／`kv-mxfp8-e4`、V620 exact `gfx1030`は
`kv-fp8-e5-block16`／`kv-mxfp8-e5`のpadded value／scale byte oracle、GPU append 6、direct attention 1、
HIP-only、fallback 0、cleanup 0をPASSした。前者はhead-dimension方向block 16、後者はstandard OCP block 32で、
いずれもtoken内E8M0 scaleを使う。標準MXFP8はexplicit-onlyでdefault候補ではない。

完全直列3 repeat品質測定のblock16 KLD p99はgfx1201 `0.0038687249522990803`、gfx1030
`0.04331390780013198`だった。一方top-1／long-contextはそれぞれ`0.85`／`0.08333333333333337`と
`0.8`／`0.16666666666666663`でfreeze済みthresholdを満たさず、両製品を`retain-fp16`とした。
early-stopによりperformance/resourceは未実行で、format correctness以外へ`project-verified`を広げない。

MI300X exact `gfx942:sramecc+:xnack-`はfresh Phase 53 reportがなく`insufficient-evidence`である。
E4M3 FNUZのblock16候補を過去Phaseの別evidenceでPASSとせず、standard OCP MXFP8はFNUZ byte列と非互換なのでunsupportedとする。
2026-08-27のユーザー指示により、この実機検証は追加のMI300X項目がまとまった時点の一括実行へ延期した。固定IPへの疎通は
VM存在またはGPU availabilityの証拠にしない。
summary `external:phase53/phase53-kv-default-summary-v3.json`のSHA-256は
`2440fd7726fca24919731abdcbd2b0f74fdd9d663ecca850b369b5ae3e69dd2b`で、詳細は
[Phase 53履歴](../history/2026/08/21-31/phase53-kv-fp8-block16-default-adoption.md)を正とする。

### 2026-08-30 current KV policy

block16 E4/E5の実装・数値結果は履歴へ固定し、production admissionを廃止した。reviewed Qwen3.5-4B BF16 dense text／
full attention／single GPU／head dim 256の省略時KVはstandard OCP MXFP8 E4M3である。V620 `gfx1030`、R9700
`gfx1201`、MI300X `gfx942:sramecc+:xnack-`を同じOCP E4M3FN value／block 32／E8M0 scale contractへ結合する。
MI300XでもFNUZへbyte reinterpretせずsoftware OCP encode/decodeを使う。明示`fp16`はrollbackである。
host/fake-HIP admissionに加え、V620 `gfx1030`とR9700 `gfx1201`のdirect GPU byte／attention oracleはPASSした。
一回のQwen3.5-4B測定ではKLD p99は両方`0.004945428206833837`、top-1一致はgfx1030 `1.0`、gfx1201 `0.85`で、
gfx1201はfreeze済み品質閾値未達である。このtarget splitを保持し、MI300X `gfx942`はfresh実機取得まで`unverified`とする。

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
