# GPU互換性方針

> 最終更新: 2026-08-24
>
> この文書はGPU対応を判定・表記する共通規則である。専用local hostのcanonical exact `gfx1030`/`gfx1201`ではformal G0/model-free G1、Phase 6のHIP VMM/production vAttention、Phase 8のBF16 Matmul/FA2-style optimized path、Phase 9のcompletion/segment・MMVF・GDN・prefill provider、Phase 15のweight NVFP4、Phase 15Oのmodel量子化最適化、Phase 15Qのmatched品質attribution、Phase 16のFP8/NVFP4 KV cacheを検証済みである。Phase 30ではexact `gfx1201`のnative FP8 readとwave-tiled causal attention、Phase 31では両targetの10,001-token chunk/arenaと明示FP8 KV経路を追加検証した。Phase 49ではGQA P32をexact `gfx1030`だけへ限定採用し、Phase 50ではexact `gfx1201`のResidual/GDN/MLP/P32候補を狭い実機scopeで検証した。各evidenceは検証した機能範囲に限定し、target全体、別SKU・別tupleへ一般化しない。

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

Phase 20では同じcanonical 2 targetで単一GGUFのQwen BF16、Gemma mixed NVFP4、Qwen MoE MXFP4をsource importerと
同じtop-1へ照合し、R9700ではQwen FP8 GGUFも実行した。全caseはexact target、HIP-only、fallbackなし、cleanup 0をPASSした。
互換性再監査後のmax-rank-4 artifactでも、再生成MoEを両target、再生成FP8をR9700で再実行した。Qwen BF16 visionは画像1枚、
233-token prefillを両targetでPASSし、R9700ではGGUF MTPとQwen GGUF serverの`/v1/models`、1-token chat、graceful
shutdownもPASSした。Qwen BF16固定laneのR9700/V620 median TTFTは46.653/184.143 ms、median TPOTは
26.689/29.685 msである。この証拠は固定artifact、single request、実行済みtargetへ限定し、性能倍率、別artifact、multi-GPUを
主張しない。

Phase 30ではexact R9700 `gfx1201`へnative E4M3FN readとwave32 causal-attention providerを限定採用した。
gfx1201/gfx1030 × FP16/FP8の各17 caseは全出力一致、fallbackなし、cleanup 0で、gfx1201の全256 E4M3FN codeも
software contractと一致した。Qwen3.5-4B BF16、4108 inputの3 process中央値はgfx1201 baseline比でTTFT 9.60%、
prefill 9.72%、E2E 9.16%、decode throughput 7.86%改善した。これは`M=1`/`M>=32`、固定model/tupleのevidenceであり、
matrix instruction、別RDNA4 SKU、別model、10000+ inputのfull-model実行を証明しない。

Phase 33ではexact gfx1030/gfx1201へFull Attentionの二つの共通限定providerを追加した。decode C1は`M=1`、
KV長1,024以上を8 waveへ分割し、N2分類を維持したユーザー承認のもと採用した。prefill C2は`M>=64`でGQA 4 headの
K/V decodeを共有し、gfx1201 N0、gfx1030 N1として採用した。4 encoding × 2 target × 29 caseは232/232 PASS、
R9700 10,000-promptとV620 4,108-promptはHIP-only、fallbackなし、cleanup 0だった。C3 matrix innerは4-row tileと
16-row WMMA shapeが合わず棄却した。この証拠を別target、別head shape、別modelへ一般化しない。

Phase 34ではexact gfx1030のQwen3.5-4B内部BF16 long-prefill projectionだけexisting hipBLASへ限定routeした。
主要5 shapeは`M>=128`、Full Attention K/V shapeは`M>=1024`で、N=32、未知shape、all-logits、短Mは旧providerを維持する。
V620 10,001-token full modelは89.249秒から34.684秒へ61.14%短縮し、R9700既存routeは不変だった。gfx1030 hipBLASLt、
別model/SKU/ROCm tuple、universal GEMM crossoverへ一般化しない。

Phase Xでは同じV620 `gfx1030`とR9700 `gfx1201`を外部llama.cpp local-subagent runtimeの比較対象にし、
Qwen3.8-27B Q5_K_XL、Q5_1 KV、context 262,144でHIP/Vulkanを実行した。HIP buildで
`GGML_CUDA_FA_ALL_QUANTS=ON`にすると、Q5_1 Flash Attention exact Qwen shapeが両target各18/18でCPU oracleへ一致し、
旧HIP build比でprefill 5.59x/11.21x、decode 4.91x/3.35xへ改善した。これはsLLM backend lifecycleを変更せず、
外部runtimeと固定tupleだけのevidenceである。

post-closeoutでは外部runtimeのmulti-GPU controlも実行し、独立V620 2 server、V620×2 layer/tensor、
R9700+V620×2 layer/tensorを比較した。独立2 serverが最大aggregate throughput、exact 524,288 context/slotを
single processで必要とする場合はV620×2 experimental tensor、3 GPUを明示的に空けられる場合はlayer split
`5,2,2`をbounded候補とした。後続のユーザー決定でV620×2 tensorを491,520 context/slotへ縮小し、parallel 2、
non-unified KVの通常local-subagent起動へ昇格した。これは外部runtimeの運用であり、sLLMのmulti-GPU support evidenceではない。
詳細は[selection summary](../../ci/matrix/phase-x-qwen38-multi-gpu-selection-v1.json)と
[local Qwen運用正本](../development/local-qwen-subagent.md)を参照する。

### 2026-08-21 Phase36 Session A: Hot Aisle MI300X VF

Session A の `project-verified` scope は、Hot Aisle の MI300X VF x1 に固定した次の exact tuple だけである。

| 項目 | 検証値 |
| --- | --- |
| GPU identity / capability | MI300X VF x1、`gfx942:sramecc+:xnack-`、wave64、304 CU、HBM `205,822,885,888` bytes、NPS1/SPX、VMM `true` |
| OS / driver | Ubuntu 24.04.4、kernel `6.8.0-124`、amdgpu `6.16.13` |
| execution toolchain | ROCm 7.14.0、HIP `7.14.60850`、LLVM 23 |
| device artifacts | logical `gfx942`、唯一のdevice bundle `gfx942:sramecc+:xnack-`、Code Object V6、ELF flags `0xE4C`（SRAM ECC on / XNACK off）、全kernel wave64（generic/別targetなし） |

operator matrix は 99/99 を PASS した。Qwen3.5-4B BF16/FP8 GGUF の固定短生成も全 dispatch が HIP-only、
fallback `0`、cleanup `0` だった。gfx942 の FP8 は OCP E4M3FN storage payload を native FNUZ resident bytes/scales へ
正しく変換し、raw reinterpret は行わない。BF16 の Hello token は `[11,353,2688,4313,310]`、FP8 は
`[11,353,1044,4313,310]` で、これは cross-provider の N1 差として記録する。Unicode と stop の確認も同じ
target/provider/cleanup contract を満たした。Phase29 由来の GDN wave32 scope leak は修正済みで、wave64 では
sequential norm を維持する。最終artifactで99/99を再実行し、BF16/FP8 resident/peakとsecond resident request、model drop後0も
確認した。終了時は foreign process `0`、全sysfs RAS blockのCE/UE `0`、VRAM baseline 復帰、provider の `/opt/rocm` link復元を確認した。

この Session A evidence は Sessions B-D、9B、low-bit KV/long context、MTP、vision、service、performance、multi-GPU、
別SKU/VM/bare-metal へ拡張しない。lifecycle は引き続き `experimental` とする。

### 2026-08-21 Phase36 Sessions B/C: MI300X long-context・MTP・vision・service

同じHot Aisle MI300X VF x1、exact `gfx942:sramecc+:xnack-`、wave64、ROCm 7.14 tupleで、Session Bは
FP16/dynamic FP8/static FP8/NVFP4 Full Attentionを各29 case、FP16 KV state 19 case、独立low-bit oracleをPASSした。
Qwen3.5-4B BF16 targetのFP16/dynamic FP8 KVはauto/512/2K/4K/8K/16Kの全12 rowが
10,001 input / 2 output、HIP-only、
fallbackなし、cleanup 0で、入力ID `23066`×10,001から生成ID `[23066,23066]`へ一致した。gfx942は
`contiguous-resident`を維持し、物理HBM/GTT観測後に共通baselineへ復帰した。VMM=trueであることだけを根拠に
`virtual-contiguous`対応とはしない。

Session CはBF16 target＋FP16 KVのMTP target-only/width 2/3/4/7/8、FP8 target＋dynamic FP8 target KVの
target-only/width 3をPASSした。MTP side pathはBF16 weights＋FP16 KVで、target側のFP8/FP8とは別にreportする。
PNG/JPEG/WebPは各64 image-pad token、同じ生成ID、first-image lazy residency、second-image reuse、shutdown cleanupを
確認した。OpenAI profile v1のraw/official client、non-stream/SSE、reasoning/stop/seed、cancel/recovery、二並行queue、
graceful shutdownもexact gfx942/HIP-only、fallbackなし、cleanup 0でPASSした。終了後はGPU process 0、provider ROCm root復元を
確認した。これはSessions B/Cの限定evidenceであり、Session Dの反復性能/llama.cpp/profile、9B、別tupleへ
一般化しない。追跡結果は[Session B summary](../../ci/matrix/phase36-mi300x-session-b-summary-v1.json)と
[Session C summary](../../ci/matrix/phase36-mi300x-session-c-summary-v1.json)に固定し、lifecycleは`experimental`のままである。

### 2026-08-21 Phase36 Session D: MI300X repeated performance

同じHot Aisle MI300X VF x1、exact `gfx942:sramecc+:xnack-`、wave64、ROCm 7.14 tupleで、4B BF16/FNUZ FP8の
short-odd、32/32、prefill-long、decode-long、10,001/2を各3+10回実行した。全10 rowはHIP-only、fallbackなし、
process終了後HBM/GTT baseline一致でPASSした。10,001/2 E2E中央値はBF16 `22.556130816`秒、FP8
`22.556528472`秒である。fixed llama.cpp peerは同じupstream model revisionだがGGUF bytes/tensor setが異なるE1で、
`0.8512540725`秒だった。rocprofv3はGDN `73.95%`、Full Attention `25.12%`を主要device familyとして分類した。
これは性能/observer evidenceであり、GPU lifecycle昇格や別tuple対応を意味しない。Sessions A〜DをPhase 36の
完了範囲としてcloseし、VMはユーザーが削除した。詳細は
[Session D summary](../../ci/matrix/phase36-mi300x-session-d-summary-v1.json)を正とする。

### 2026-08-21 R9700 direct 10,001/2 E2E comparison

canonical R9700 exact `gfx1201`で、Qwen3.5-4B BF16、FP16 KV、`23066`×10,001 input / 2 output、greedy、
3 warmup＋10 measuredをcurrent-source sLLMとfixed llama.cpp `b10453`へ実行した。E2E中央値は
sLLM `3.936429665`秒、llama.cpp `2.063845785`秒で比`1.90733x`、prefill比は`1.89197x`だった。
生成は全試行`[23066,23066]`、sLLMはHIP-only、fallback/cleanup 0、llama.cppは33/33 layer full offload、
終了後process 0、VRAM 0%へ復帰した。model GGUF bytes/tensor setが異なるためE1 system-equivalentに限定し、
別SKU/model/shape/ROCm、strict artifact identity、一般性能保証へ拡張しない。
[R9700 summary](../../ci/matrix/r9700-sllm-llama-e2e-v1.json)を正とする。

### 2026-08-21 Phase40 token selector・grammar scope（verification中）

Phase 40では、backend-neutral sampler chain、bounded GBNF/JSON Schema、raw-byte/token-trie/partial UTF-8 state、post-mask logprobs、
`n=1..=8` choice state、strict OpenAI wireをhostへ実装した。HIP側は既存Argmax ABIを変更せず、additive `TokenSelect` contract、
Qwen/Gemma terminal adapter、F32 additive/U8 valid-mask、completion後の固定16-byte selected record readbackを追加した。
このscopeではfull-vocabulary logits D2Hを行わず、GPU selector非対応のsamplerはhost pathへ明示routeする。

| target / scope | lifecycle | evidence | status |
| --- | --- | --- | --- |
| V620 exact `gfx1030` Phase40 selector contract matrix | `experimental` | `project-verified`（selector scope） | vocab `1,3,17,255,256,257,248320`×counter `0,1`、CPU token/logprob、fallback 0、selected-only D2H 16 bytes |
| R9700 exact `gfx1201` Phase40 selector contract matrix | `experimental` | `project-verified`（selector scope） | V620と同一matrix。既存10,001/2 E1 E2Eはselector matrixと別 |
| MI300X exact `gfx942:sramecc+:xnack-` Phase40 route/compile | `experimental` | `unverified` | wave64 feature-pinned compile-only PASS。real correctness/performanceはdeferred |

V620/R9700 selector matrixはodd vocabulary/mask/bias、NaN/Inf/all-mask、fixed seed、CPU token/logprob oracle（tolerance `.005`）、
fallback 0、selected record D2H 16 bytes、full-vocabulary D2H 0を記録済みである。GPU unavailable、timeout、crash、zero selectionは
PASSにしない。Qwen/Gemma sampled-generationの最終統合runは別途継続する。Phase40詳細は
[archive plan](../plans/archive/2026/08/21-31/phase40-token-selection-grammar-structured-generation.md)と
[history](../history/2026/08/21-31/phase40-token-selection-grammar-structured-generation.md)および
[tracked GPU summary](../../ci/matrix/phase40-token-selector-gpu-summary-v1.json)を正とする。Phase40ではllama.cpp sourceの
直接reuseはなく、既存provenance lockを変更しない。

### 2026-08-22 Phase41 opaque state fork/image scope

Phase 41はprefix/context/checkpointが共有するopaque KV/linear state fork、COW、全plane export/importをadditive public HIP ABIへ
実装した。focused real-GPU runnerはmodel-free state contractを独立oracleへ照合し、full-model性能や未実行model/providerへ一般化しない。

| target / scope | lifecycle | evidence | status |
| --- | --- | --- | --- |
| V620 exact `gfx1030` Phase41 state matrix | `experimental` | `project-verified`（state scope） | FP16 VMM fork/COW 63/64/65/127/128/129、FP8 dynamic/static、NVFP4、linear 5 planes、fallback/cleanup 0 |
| R9700 exact `gfx1201` Phase41 state matrix | `experimental` | `project-verified`（state scope） | V620と同一matrix、source/child byte exact、target-only、fallback/cleanup 0 |
| MI300X exact `gfx942:sramecc+:xnack-` Phase41 route/compile | `experimental` | `unverified` | ROCm 7.14/LLVM 23、wave64 feature-pinned compile-only PASS。real run deferred |

FP16 child append後もsource K/Vは不変で、encoding別2/4/6 planeとlinear active slot/scratchをbyte/state exactに照合した。
両実機run後のGPU process、cleanup failure、uncorrectable ECCは0だった。Qwen/Gemma production checkpoint/contextのfull-model
実機一般対応をこのmodel-free matrixだけから推論しない。詳細は
[Phase41 GPU summary](../../ci/matrix/phase41-state-gpu-summary-v1.json)、
[archive plan](../plans/archive/2026/08/21-31/phase41-prefix-session-speculation.md)、
[history](../history/2026/08/21-31/phase41-prefix-session-speculation.md)を正とする。

### 2026-08-24 Phase49 exact `gfx1030`限定採用

Phase 49はcanonical V620 exact `gfx1030`（UUID `GPU-76a08c022586fed6`、BDF `0000:03:00.0`）の性能候補だけを対象にし、GQA4 decodeのP32 partitionを
KV長4,096以上、head dimension 256、FP16 KV、`M=1`へ限定して既定採用した。long-prefill v2とHIP Graphは
既定経路へ採用せず、R9700 `gfx1201`とMI300X `gfx942`へこのselector、閾値、solution ID、wave32 binaryを
自動展開しない。最終通常5行の正しさ・資源・HIP-only・cleanupを確認したが、長行を含む全7行同等は主張せず、
未達行は後続Phaseの入力として保持する。lifecycleは`experimental`、evidenceは当該V620 scopeだけを
`project-verified`とする。詳細は[Phase 49/50数値台帳](numerical-output-changes.md)と
[Phase 49以降ロードマップ](../history/2026/08/21-31/phase37-plus-mi300x-and-llama-gap-roadmap.md)を参照する。

### 2026-08-24 Phase50 exact `gfx1201`狭い実機scope

Phase 50でGPU性能を実証したのはcanonical R9700 1台のexact `gfx1201`だけである。UUID、BDF、HIP targetを
相互照合した固定identityは、UUID `GPU-a8e9ddefa2d60f55`、BDF `0000:07:00.0`、`gfx1201`である。
Residual RMSNorm、GDN projection、MLP gate/up/SiLU、GQA P32のA/B比較を、固定Qwen3.5-4B BF16、
FP16 KV、`M=1`、単一requestのtarget専用selectorで確認し、最終通常行と長行の計測・未達判定をこのscopeへ限定して
記録した。100,000-token prefillはOOMで完走せず、`project-verified`の成功scopeへ含めない。20,000-token decodeの
完走結果は[Phase 50履歴](../history/2026/08/21-31/phase50-r9700-port-and-mi300x-handoff.md)とsummaryへ固定し、この方針文書では数値を重複しない。
candidate分類と未達理由は履歴および[数値変更台帳](numerical-output-changes.md)を正とする。

| target / scope | lifecycle | evidence | status |
| --- | --- | --- | --- |
| R9700 exact `gfx1201` Phase 50 Residual/GDN/MLP/P32 A/B、通常・長行 | `experimental` | `project-verified`（上記scopeのみ。100k OOM未達は除外） | HIP-only、fallbackなし、selector/cleanupを確認。別shape・別model・別SKU・別tupleへ一般化しない |
| MI300X logical `gfx942` / feature付きdevice target `gfx942:sramecc+:xnack-` Phase 50 handoff | `experimental` | `unverified`（compile/host scope） | Code Object V6、wave64、`sramecc=on`/`xnack=off`のcompile/linkとhost selector非選択のみ。実機runtime/PASSはPhase 51待ち |

Phase 50の固定local tupleはUbuntu 24.04.4、kernel `6.17.0-35-generic`、amdgpu `6.16.13`、ROCm 7.14.0、
HIP `7.14.60850`、LLVM 23、Code Object V6、wave32である。R9700 buildはexact `gfx1201`専用とし、genericまたは
multi-arch artifactを性能evidenceへ使わない。MI300Xは論理target `gfx942`、feature付きdevice/codegen target
`gfx942:sramecc+:xnack-`、wave64、Code Object V6へ固定する。production Cargoの`CMAKE_HIP_ARCHITECTURES`はlogical `gfx942`を使い、
feature suffix付きtargetはdirect CMake probeだけで扱う。gfx1201 providerがhost selectorで選択されないことを示す。このcompile/link結果は既存Hot Aisle MI300X runtime evidenceを
拡張せず、Phase 51のfresh実機検証を待つ。Phase 50のいずれのscopeもlifecycleを`supported`へ昇格させず、
RDNA4全体、CDNA3全体、他のOS・driver・ROCm・SKUへ互換性を推論しない。

### 2026-08-24 Phase52 R9700 100k resident KV scope

canonical R9700 UUID `GPU-a8e9ddefa2d60f55`、BDF `0000:07:00.0`、exact `gfx1201`で、Qwen3.5-4B BF16、
FP16 KV、単一requestの`100,000/2`を自動prefill 2,048、1 warmup＋3 measuredで4/4 PASSした。
logical capacity 131,072は`contiguous-resident`、8 KV layerのK/V commitは4 GiBである。生成は全て
`[23066,23066]`、HIP-only、fallback/cleanup failure 0、process消滅、HBM/GTT baseline復帰を確認した。
`10,001/2`も短い`virtual-contiguous`経路で13/13 PASSしたため、resident選択を65,536未満へ広げない。

この結果はPhase 50の100k OOM履歴を削除せず、Phase 52 source/binary/tupleの追加成功scopeとして扱う。R9700全model、
RDNA4全SKU、別driver/ROCm、batch/parallel、Paged Attentionへ一般化せず、lifecycleは`experimental`のままとする。
全反復とidentityは[Phase 52 summary](../../ci/matrix/phase52-r9700-kv-commit-summary-v1.json)を正とする。

| target / scope | lifecycle | evidence | status |
| --- | --- | --- | --- |
| R9700 exact `gfx1201` Phase 52 Qwen3.5-4B BF16 `100,000/2` | `experimental` | `project-verified`（固定単一request scope） | 自動2K、resident KV、4/4 PASS、生成一致、HIP-only、fallback/cleanup 0、資源復帰 |

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
