# AMD GPU互換性方針

> 最終更新: 2026-08-03
>
> この文書はAMD向けの識別規則と初期候補を記録する。現時点の初期targetはすべて`lifecycle=experimental`である。計画targetのevidenceは`unverified`、canonical local実機のformal model-free G0/G1は検証した限定範囲だけ`project-verified`とする。

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

uLLMは次の二経路を分ける。

- generic path: 配布binary数を抑えるbaseline。対象processor集合、`generic_processor_version`、feature stateを照合する。
- exact fast path: `gcnArchName`とfeature stateが一致するexact target向け。固有命令とtarget別resource tuningを利用する候補とする。

generic processor versionをELF `e_flags`へ保持する仕組みは[Code Object V6以降](https://llvm.org/docs/AMDGPUUsage.html#amdgpu-elf-header-e-flags-table-v6-onwards)に定義されるため、uLLMのgeneric pathはCode Object V6以降に限定する。基準toolchainのROCm 7.14.0同梱compilerが意図したgeneric version/feature stateを生成し、同releaseのROCr loaderが初期対象実機で受理することは今後検証する。検証完了まではgeneric pathも`evidence=unverified`である。

generic binaryへ黙ってfallbackしてよいとは限らない。特に`gfx9-4-generic`は`gfx942`と`gfx950`を覆う一方、LLVM表ではFP8/BF8命令と変換命令が利用不可であるため、初期CDNA 3 FP8 pathには採用しない。

## 初期AMD target集合

次は初期実装の計画範囲であり、現在の動作実績ではない。

| 表示分類 | exact target | generic baseline | lifecycle | evidence |
| --- | --- | --- | --- | --- |
| RDNA 2 | `gfx1030`–`gfx1036` | `gfx10-3-generic`、Code Object V6+ | `experimental` | `unverified` |
| RDNA 4 | `gfx1200`, `gfx1201` | `gfx12-generic`、Code Object V6+ | `experimental` | `unverified` |
| CDNA 3 | `gfx942` | 初期FP8 pathでは使用しない | `experimental` | `unverified` |

[ROCm 7.14.0 release notes](https://rocm.docs.amd.com/en/docs-7.14.0/about/release-notes.html)は、ROCm componentと公式対象製品をexact target単位で掲載する。generic processorのcoverage、compilerがtargetを生成できること、製品・OS構成のvendor supportは別の事実である。

## FP8の根拠とinterop contract

FP8はhardware ISA evidenceとlibrary/uLLM contractを分ける。

### Hardware ISA evidence

AMDのROCm 7.14.0[data types and precision support](https://rocm.docs.amd.com/en/docs-7.14.0/reference/precision-support.html)は、CDNA 3 matrix coreのnative FP8をFNUZ、RDNA 4のFP8をOCP系として区別している。[gfx942 instruction syntax](https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/AMDGPU/AMDGPUAsmGFX940.html)にもFP8/BF8 matrix命令が列挙される。これらをhardware capability mappingの根拠とする。

- `gfx942`（CDNA 3）: native FNUZ系FP8（E4M3 FNUZ/E5M2 FNUZ）。
- `gfx1200`/`gfx1201`（RDNA 4）: OCP系FP8（E4M3/E5M2）。
- `gfx1030`–`gfx1036`（RDNA 2）: native FP8 matrix pathなし。将来FP8 modelを扱う場合は変換またはemulationを別pathとして明示する。

### uLLMとhipBLASLtの初期interop contract

[ROCm 7.14.0 release notes](https://rocm.docs.amd.com/en/docs-7.14.0/about/release-notes.html)は同releaseのhipBLASLtを1.4.1とする。そのversionの[hipBLASLt precision support](https://rocm.docs.amd.com/projects/hipBLASLt/en/docs-7.14.0/reference/data-type-support.html)は、同表がlibrary自身のdatatype supportでありhardware supportを示さないと明記している。

uLLMがhipBLASLt 1.4.1のFP8 GEMM pathを使用する場合の初期contract候補は次とする。

- exact `gfx942`: `hipblaslt_f8_fnuz`/`hipblaslt_bf8_fnuz`を使う。
- exact `gfx1200`/`gfx1201`: `hipblaslt_f8`/`hipblaslt_bf8`を使う。
- model storage encoding、uLLM kernel input、hipBLASLt datatypeが異なる場合は明示的に変換し、FNUZとOCP payloadを再解釈しない。

このcontractは初期FP8実装の計画であって、現在の実機検証結果ではない。hipBLASLt表だけをhardware全体の対応根拠にせず、hardware mappingとlibrary queryの両方を満たしたproblemだけdispatchする。

## 製品別evidence

2026-08-03時点の事実を次のように記録する。すべて`lifecycle=experimental`であり、evidenceはupstream掲載、local実機G0/G1、未検証範囲を分けて記載する。

| 対象構成 | upstreamの事実 | evidence |
| --- | --- | --- |
| `gfx942`のROCm 7.14.0掲載MI300製品・OS構成 | AMDの現行support資料に掲載 | `[vendor-supported, unverified]` |
| `gfx1200`/`gfx1201`のROCm 7.14.0掲載RDNA 4製品・OS構成 | AMDの現行support資料に掲載 | `[vendor-supported, unverified]` |
| `gfx1030`のROCm 7.14.0掲載Radeon PRO W6800/V620製品・OS構成 | AMDの現行support資料に掲載 | `[vendor-supported, unverified]` |
| canonical Radeon AI PRO R9700 1台、exact `gfx1201` | Ubuntu 24.04.4、kernel `6.17.0-35-generic`、amdgpu `6.16.13`、ROCm 7.14.0でformal model-free G0/G1を実行 | `[project-verified]` |
| canonical Radeon Pro V620 1台、exact `gfx1030` | 同じlocal host tupleでformal model-free G0/G1を実行。2台目V620はspareであり必須evidence外 | `[project-verified]` |
| consumer RDNA 2 / RX 6000系 | `gfx1030`の掲載だけでは同じtargetを持つ全consumer SKUの公式対応にならない。[Radeon Linux support matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/native_linux/native_linux_compatibility.html)に完全なSKU/OS構成が掲載された場合だけ`vendor-supported`を付ける | `[unverified]` |
| `gfx1031`–`gfx1036` | LLVMのcode target定義は製品構成のvendor supportではない | `[unverified]` |

local `project-verified` の範囲は、commit `f393d688a051d2b73c8773d8a930a711592609bc`のcanonical exact `gfx1030`/`gfx1201` artifactに対するG0 identity/healthと、1、3、17、255、256、257 byteのG1 allocation、copy、diagnostic dispatch、completion、byte-exact copy-backだけである。Code Object V6、wave32、target別ELF flags、artifact metadata、実loader pathを検証したが、capability profile、resource gate、semantic数値kernel、model、性能は未検証であり、target全体や別SKUへ一般化しない。完全なsoftware tupleと実行結果は[software compatibility](software.md)に記録する。

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
