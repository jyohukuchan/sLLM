# GPU互換性方針

> 最終更新: 2026-08-16
>
> この文書はGPU対応を判定・表記する共通規則である。専用local hostのcanonical exact `gfx1030`/`gfx1201`ではformal G0/model-free G1、Phase 6のHIP VMM/production vAttention、Phase 8のBF16 Matmul/FA2-style optimized path、Phase 9のcompletion/segment・MMVF・GDN・prefill provider、Phase 15のweight NVFP4、Phase 15Oのmodel量子化最適化、Phase 15Qのmatched品質attribution、Phase 16のFP8/NVFP4 KV cacheを検証済みである。各evidenceは検証した機能範囲に限定し、target全体、別SKU・別tupleへ一般化しない。

## 二層の識別モデル

GPU対応をSKU名やarchitectureの通称だけで決めない。sLLMは`binary_key`と`capability_profile`を分離する。

### `binary_key`

ロード・実行できるbinaryを一意に選ぶためのキーで、少なくとも次を含む。

```text
(backend, execution ABI, code target, codegen features)
```

- backend: HIP、CUDAなど。
- execution ABI: code object、driver/runtime、host platformに関わるABI。
- code target: AMDのexact `gfx`/generic target、NVIDIAの`sm_XX`/PTX targetなど。
- codegen features: AMDの`xnack`/`sramecc`/wavefront size、NVIDIAのarchitecture-specific suffixなど、ロード可否または命令生成を変えるfeature。

NVIDIAの[cubinとPTXの互換規則](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html#application-compatibility-on-the-nvidia-blackwell-gpu-architecture)が示すように、native binaryとJIT可能な中間表現は同じ互換性を持たない。AMDでもexact targetとgeneric code objectを同一視しない。

### `capability_profile`

選択したbinary上で、kernelを安全かつ有用な速度で実行できるかを判定するprofileである。

- matrix engineの有無、世代、利用可能な命令形。
- FP4、FP8 encoding、BF16、FP16、INT8などのprecisionとaccumulationの組み合わせ。
- warp/wave sizeと同期・memory model上の前提。
- total device-local memory、memory bandwidth、cache/LDS/shared memory、registerなどのresource。

同じcode targetでもcapabilityは同じとは限らない。反対に似たcapabilityを持っていてもbinary ABIまたはcode targetが違えばbinaryは共有しない。

## Capability情報の取得

全capabilityを一つのruntime APIから直接取得できるとは仮定しない。出所とversionを保持し、次の順で組み立てる。

1. runtime APIから正規target、codegen feature、wave/warp、memory量など取得可能な値を読む。
2. matrix engine、precision、公称resourceなどruntimeで直接得られない値を、vendor device IDとexact targetをキーにしたversioned vendor mappingで補う。
3. library固有pathは、使用中のlibrary versionが提供するcapability/solution queryで確認する。
4. 文書やqueryだけで確定できないkernel capabilityは、小さなcapability probeで確認し、driver/runtime/library versionとともに結果をcacheする。

vendor mapping、library query、probeは代替関係ではない。例えばhardwareが命令を持つことは特定libraryのdatatype contractを保証せず、libraryがdatatypeを列挙することもhardware ISA全体の証拠にはならない。

## Resource gate候補

次は採用済みの互換性契約ではなく、要件資料にある**未確定の初期候補**である。

> INT8とFP16またはFP4で1TOPS以上、専用メモリ16GB以上、理論帯域250GB/s以上、普及度

判定値を混同しない。

| 値 | 用途 |
| --- | --- |
| total device-local memory | `専用メモリ16GB以上`というeligibility候補に使う。起動時の空き容量ではない |
| 起動時free memory | modelとruntime workspaceを収容できるかという動的admissionに使う。GPU自体のlifecycleを変更しない |
| vendor theoretical bandwidth | vendorが公表した理論帯域。`理論帯域250GB/s以上`候補の第一資料とする |
| derived theoretical bandwidth | 公表値がない場合にmemory data rate、bus width、channel数等から導出し、式と入力資料を残す |
| measured bandwidth | 実効性能の観測値。現段階では理論帯域gateの代替にしない |

次は未解決であり、解決前にresource gate不合格だけを理由として`unsupported`へ確定しない。

- `INT8とFP16またはFP4`の論理式、1 TOPSを各precisionへどう適用するか、sparse値を許すか。
- `16GB`の境界をdecimal GBのまま扱うか、multi-die/partitioned memoryをどう数えるか。
- unified memoryを持つTegra/APUを「専用メモリ」候補へどう対応付けるか。
- vendor公表値がない場合のderived theoretical bandwidthの標準式と、測定値による例外を許すか。
- 普及度の指標、判定時点、例外の承認基準。

## 状態の二軸

状態は一つのtierへ合成せず、`lifecycle`と`evidence`を別々に記録する。対象は`binary_key + capability_profile + OS + driver/runtime/library`の具体的な構成である。

### `lifecycle`

| 値 | 意味 |
| --- | --- |
| `supported` | sLLMが互換性契約として受け入れ、不具合修正対象とする |
| `experimental` | 初期対象として実装・bring-up中、または試験的に利用可能だが互換性契約には未昇格 |
| `planned` | 将来対応の意図はあるが、現在の実装対象または利用可能pathではない |
| `unsupported` | 対象外と明示決定した、resource/capability要件を確定的に満たさない、または既知の非互換 |

### `evidence`

| 値 | 意味 |
| --- | --- |
| `vendor-supported` | vendorのsupport matrixが対象SKU、OS、driver/runtime/libraryの構成を掲載 |
| `project-verified` | sLLMが同じ具体的構成と対象機能を実機検証済み |
| `unverified` | sLLMによる該当構成・機能の実機検証結果がない |

`evidence`は必要なら複数付ける。例えばvendor公式構成をsLLMがまだ検証していなければ`[vendor-supported, unverified]`となる。同じ構成・機能をsLLMが検証した時点で`unverified`を外し、`project-verified`を付ける。toolchainがtargetを生成できることだけでは`vendor-supported`でも`project-verified`でもない。

`lifecycle=supported`への昇格には原則として同じscopeの`project-verified`を要求する。`vendor-supported`だけで自動昇格せず、反対にvendor公式範囲外でも十分なproject evidenceがあれば`project-verified`を保持できる。

現時点の初期AMD targetは`lifecycle=experimental`である。canonical V620 `gfx1030`とR9700 `gfx1201`は
model-free G0/G1に加え、Phase 6 A0のHIP VMM primitive、A1のFA2-style proxy比較と
virtual-contiguous KV最小production pathについて`project-verified`である。他の機能scope、SKU、tupleは
引き続き`unverified`を含む。AMDの製品別状態と検証scopeは[AMD GPU互換性方針](amd-gpu.md)、
KV方式の範囲は[KV memory decision](../architecture/kv-memory.md)に記録する。

Phase 6 A6では同じcanonical 2 targetを各UUIDで単独可視化し、Qwen3.5-4BのOpenAI-compatible
non-stream/SSE/stop/disconnect service pathを追加検証した。これは単一GPU・単一model-resident serverの
限定scopeだけを`project-verified`とする。複数GPU可視のprocess、global physical indexによるworker選択、
multi-GPU servingは検証範囲外である。

Phase 7ではcanonical V620/R9700のexact tupleをversioned recordに固定し、`gfx1030`〜
`gfx1036`、`gfx1200`、`gfx1201`、`gfx942`の10 targetをROCm 7.14.0でcompile-only検査する
lifecycle profileを追加した。この10-target結果はcode object生成だけの証拠であり、
canonical `gfx1030`/`gfx1201`以外の実機、SKU、software tupleを`project-verified`へ昇格しない。

Phase 9では同じcanonical 2 targetでkernel-only/hipBLAS mixed HIP Graph PoC、BF16 M=1 MMVF v3、
target別GDN state layout、completion segment、4B direct engineを追加検証した。short-odd中央値はV620で
TTFT/E2E 0.306/0.855秒、decode 29.69 tok/s、R9700で0.051/0.490秒、37.20 tok/sだった。
これはexact target、Qwen3.5 BF16 single request、該当shape/providerの`project-verified` evidenceである。
production全体のHIP Graph replay、別SKU、multi-request、別software tuple、長時間安定性は証明しない。

Phase 12ではHot Aisle MI300X VF x1の完全tupleで、feature付き実device名
`gfx942:sramecc+:xnack-`をexact logical `gfx942` artifactへfail-closedに対応づけ、wave64 BF16、FNUZ FP8、
contiguous-resident KV、4B/9B model、OpenAI service、4B performanceを実機検証した。このscopeを
`project-verified`とするが、MI300A/MI325X、別MI300X VM/image、bare metal、multi-GPU、generic `gfx9-4`、
未実行model/shapeへ一般化しない。VMM=trueでもPhase 12の固定比較条件としてgfx942だけresident providerを明示選択した。

Phase 16Fではcanonical V620 `gfx1030`とR9700 `gfx1201`でNVFP4 dynamic-W4A4 operator、static-FP8 KV、
Unsloth Gemma 4 12B mixed full graphを実機検証した。この限定scopeは`project-verified`だが、same-artifact NVIDIA reference
runtimeを未実行のためmodel evidence lifecycleは`experimental`である。NVIDIA Gemma 4 31B NVFP4とKimi K3 MXは
metadata/encoding handoffのみであり、AMDまたはNVIDIA full-model execution evidenceではない。

Phase 17では同じcanonical 2 targetでQwen3.5-4B MTP real-weight componentとvision 297 tensorを実行した。
visionは256 patch token/64 visual token、multimodal text prefill+decode、deterministic replay、HIP-only、fallbackなし、cleanup 0を
両targetでPASSした。MTPもdraft `[198,248044]`、target verify `[198,248045,248045]`、accepted 1で一致したが、現行の
逐次verifyはtarget forward数を減らさないため通常providerはtarget-onlyを維持する。このevidenceは固定model/image geometry、
single request、BF16 visionに限り、別SKU、video、low-bit vision、性能優位性へ一般化しない。

Phase 18では同じcanonical 2 targetでQwen3.5-4Bのserial-equivalent M=2/3/4/7/8 target verifyを実行した。BF16+FP16 KVと
FP8 W8A8+static FP8 KVでtoken/hidden、M=8 raw logits、accepted-prefix K/V payloadを逐次M=1へbit/byte exact照合した。
R9700 `gfx1201`のBF16 fixed 32-token caseはMTP off/on中央値`1.0355x`でnoiseを越えたため通常内部providerへ採用し、V620
`gfx1030`はwidth 1中央値`0.9990x`でnoise内のためtarget-onlyを維持する。このevidenceはfixed model、single request、text-only greedy、
実行済みtuple/lengthへ限定し、別model/SKU、sampling高速化、一般的なMTP倍率へ一般化しない。

Phase 19では同じcanonical 2 targetで`amd/Qwen3.5-35B-A3B-MXFP4` revision `2e19c657...`のsingle-GPU
text-only sparse MoEを実行した。actual-weight oracleはlayer 0/19/39、M=1/3/7、連続8 expertで最大絶対・相対誤差
`1.86265e-9`、fallback 0をPASSし、full modelは40 SparseMoe submissionとprefill/decode 960/320 active pairを
exactに記録した。resident/peakは22,009,574,016/22,230,758,892 byteである。R9700 `gfx1201`のprefill/decode中央値は
216.258/204.198 ms、V620 `gfx1030`は537.832/370.711 ms（各2 warmup + 11 measured）だった。
CLI/OpenAI non-stream/SSE/cancel/recovery/seed/shutdownも通常経路でHIP-only、fallbackなし、cleanup 0をPASSした。
このevidenceはfixed artifact、batch 1、text-only、実行済みtupleへ限定し、vision/MTP、multi-GPU、別model/SKUへ一般化しない。

### software.mdとの関係

[ソフトウェア互換性方針](software.md)も完全なsoftware tupleのlifecycleを`supported`、`experimental`、`planned`、`unsupported`の四値に統一する。実機検証はsoftware lifecycleではなく、完全なtuple、日時、結果、対象機能を残す検証history/evidenceである。対象GPU機能まで同じtupleで検証した履歴は`evidence=project-verified`を支え、lifecycleを`supported`へ変更する根拠になり得る。

GPU evidenceが十分でも、OS、runtime library、artifact条件まで一致しなければsoftware tupleは`supported`にならない。また、software lifecycleが`experimental`であることからvendor公式対応の有無を推論しない。softwareとGPUの状態から直積を推論せず、完全な構成のlifecycleとevidenceを別fieldとして記録する。

## 現在と将来の範囲

| 対象 | lifecycle | evidence | 備考 |
| --- | --- | --- | --- |
| RDNA 2、RDNA 4、CDNA 3 | `experimental` | `unverified`を含む | 初期AMD対象。製品別vendor evidenceはAMD文書を参照 |
| RDNA 3、RDNA 3.5 | `planned` | `unverified` | 将来AMD候補 |
| MI50、CDNA 1、CDNA 2、CDNA 4、CDNA 5 | `planned` | `unverified` | 将来AMD候補 |
| NVIDIA backend | `planned` | `unverified` | 将来候補。以下の分類例は現在対応の宣言ではない |
| CPU backend | `planned` | `unverified` | GPUではないが同じbackend状態体系で管理する将来候補 |

未列挙targetは未分類であり、自動的に`unsupported`とはしない。`unsupported`は対象外または非互換を明示決定したときだけ付ける。

## NVIDIAを使った将来判定例

以下は二層モデルが必要な理由を示す将来の分類例である。NVIDIAは[compute capabilityをhardware featureとinstructionの識別子](https://developer.nvidia.com/cuda-gpus)としているが、それだけで製品ごとの全能力とresourceは表せない。

| binary/code target class | capability profileで分ける例 | 判定上の要点 |
| --- | --- | --- |
| Turing `sm_75` | GeForce GTX 16系: Tensor Coreなし。GeForce RTX 20系: Tensor Coreあり | [legacy CC表](https://developer.nvidia.com/cuda/gpus/legacy)では両系列がCC 7.5だが、[GTX 16比較表](https://www.nvidia.com/en-eu/geforce/graphics-cards/compare/?section=compare-16)はTensor Coreを`-`とし、[RTX/GTX公式解説](https://blogs.nvidia.com/blog/whats-the-difference-between-nvidia-rtx-and-gtx/)はRTX側のTensor Coreを説明する。従ってTensor Core accelerated pathは共有できない。CUDA-core fallbackの可否は必要precision、性能、resourceをkernel別に判定する |
| Ampere `sm_80` class | A100等 | [`nvcc` target一覧](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list)に従ってclassを分け、class内でもSKU resourceをprofile化する |
| Ampere `sm_86` class | RTX 30/A40等 | `sm_80` classのcapabilityを流用せずkernel要件を照合する |
| Ampere `sm_87` class | Jetson Orin等 | Tegraの統合memoryとhost platform ABIをdesktop/datacenterから分ける |
| Blackwell `sm_100`, `sm_103`, `sm_110`, `sm_120`, `sm_121` | B200/GB300、Jetson Thor、RTX、GB10等 | CUDAの[`nvcc` target一覧](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list)に従い、`a`/`f` suffixも`binary_key`へ含める |
| Tegra: Xavier `sm_72`、Orin `sm_87`、Thor `sm_110` | unified memory、GPU利用可能量、帯域を個別profile化 | host platform ABIは`binary_key`、resourceはprofileに置く。旧製品のCCは[NVIDIA legacy GPU表](https://developer.nvidia.com/cuda/gpus/legacy)、現行製品は[現行GPU表](https://developer.nvidia.com/cuda-gpus)を正とする |

BlackwellのThor targetはCUDA 12.9以前の`sm_101`からCUDA 13.0以降の`sm_110`へ改名されている（NVIDIAの[cuRANDDx release notes](https://docs.nvidia.com/cuda/curanddx/0.2.1/release_notes.html)）。toolchain versionを無視して数値だけを永続キーにしない。また、[Blackwell compatibility guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/index.html#application-compatibility-on-the-nvidia-blackwell-gpu-architecture)は`sm_100a`のような`a` targetがforward/backward compatibleでないことを明記している。

## 互換表の記載規則

- vendor資料に掲載されたtarget、sLLMの実装有無、実機検証結果という事実を分ける。
- 初期候補と将来候補は計画として記し、現在の動作実績と混ぜない。
- 未検証は`evidence=unverified`と明記し、`works`や`verified`と断定しない。
- 変化するvendor support matrixには確認日と公式一次資料へのリンクを付ける。
- lifecycleとevidenceの変更履歴を残し、過去の検証記録を消さない。
