# ソフトウェア互換性方針

> 最終更新: 2026-08-24

## 目的

この文書は、sLLM のビルドおよび実行に使う OS、ツールチェーン、ROCm の互換性契約を定義する。ここに記すバージョンは初期決定であり、実装や実機検証で問題が判明した場合は、コードだけを回避的に変更せず、この文書の tuple と判断理由も更新する。

## 基準ツールチェーン

| 項目 | 初期決定 | 方針 |
| --- | --- | --- |
| OS | Ubuntu 24.04 LTS | 主開発・主配布環境。point release と kernel も compatibility tuple に記録する |
| Rust edition | 2024 | workspace 全体で統一する |
| Rust MSRV | 1.85.0 | `rust-version = "1.85.0"` として公開クレートに明記する |
| Rust 開発 pin | 1.97.1 | 2026-08-02 時点の開発用 toolchain。`rust-toolchain.toml` で固定する |
| Cargo resolver | 3 | virtual workspace の `[workspace]` に `resolver = "3"` を明記する |
| Cargo lockfile | commit する | sLLM は application であるため、workspace root の `Cargo.lock` を version 管理する |
| C++ | C++17 | `native/hip` の host code と HIP translation unit に共通して要求する |
| ROCm | 7.14.0 | 同一 ROCm release から compiler、runtime、headers、libraries を揃える |
| HIP compiler | ROCm 7.14.0 同梱 `amdclang++` | system LLVM ではなく、選択した ROCm tree の compiler を使う |
| LLVM | 23 | ROCm 7.14.0 に含まれる LLVM 系列 |
| CMake | 3.21 以上 | `cmake_minimum_required(VERSION 3.21)` とする |
| Python host CI | 3.12.10 | H0〜H2のtest harnessとNumPy oracle専用。production runtimeの実装言語・互換性契約ではない |

Rust 2024 は Rust 1.85.0 で安定化されたため、edition と MSRV を 1.85.0 に揃える。resolver 3 は Rust version を考慮する resolver であり、virtual workspace では edition から暗黙に決まらないため明示する。開発 pin は再現性のための固定値であって MSRV ではない。依存クレートを選ぶ際は、開発 pin でビルドできるだけでなく MSRV を超えないことを要求する。

Python host CIのtransitive dependencyを含むexact versionとartifact SHA-256は`ci/requirements-host.txt`、実行入口とCPU-only境界は`docs/development/testing.md`を正とする。dependency取得とtest実行を分け、required host testはnetwork namespaceで外部接続を遮断し、model cache、GPU fallbackを使用しない。

## ROCm の発見と一貫性

ビルド・configure 全体の契約では、CMake に明示された `ROCM_PATH`（例: `-DROCM_PATH=...`）を最優先とする。明示指定した root が存在しない場合、または検査に失敗した場合は明示的に失敗させ、環境変数や既定 rootへ fallback しない。

開発環境の `scripts/dev/activate-rocm.sh` は、ROCm root を次の優先順位で一つだけ選ぶ。

1. 環境変数 `SLLM_ROCM_PATH`
2. スクリプト実行前から定義されている環境変数 `ROCM_PATH`
3. 標準配置 `/opt/rocm/core-7.14`

`SLLM_ROCM_PATH` またはスクリプト実行前から定義されている `ROCM_PATH` が定義されて空の場合は明示的に失敗させる。選択した root が存在しない場合、または検査に失敗した場合も明示的に失敗させ、既定 rootや別 releaseへfallbackしない。

選択後は path を canonicalize し、compiler、HIP headers、CMake package、runtime、device libraries をすべてその root から解決する。HIP compiler は原則として安定した entry point `${ROCM_PATH}/bin/amdclang++` を使い、その symlink を解決した実体も同じ ROCm root 内にあることを検査する。package manager 配置と tarball 配置で LLVM の内部 directory が異なるため、`${ROCM_PATH}/llvm` または `${ROCM_PATH}/lib/llvm` を無条件に仮定しない。発見した root、ROCm release、`amdclang++ --version` の LLVM major を configure 時に検査し、ROCm 7.14.0、LLVM 23、または期待する配置と一致しない場合は明示的に失敗させる。暗黙に system `clang++`、別の `/opt/rocm-*`、別 release の library へフォールバックしてはならない。

「ROCm components は同一 release」とは、各 component 固有の内部バージョン番号を `7.14.0` に揃えるという意味ではない。ROCm 7.14.0 の配布物・repository として組み合わせて公開された compiler、HIP runtime、ROCr、math libraries、headers、device libraries を混在させずに使う、という意味である。

### GPU target と codegen feature

HIP binary の target は host の自動検出結果だけで決めず、Cargo から CMake へ `CMAKE_HIP_ARCHITECTURES` を明示的に渡す。`xnack`、`sramecc`、wavefront size など、binary compatibility または命令生成を変える codegen feature は project 固有の `SLLM_HIP_CODEGEN_FEATURES` に正規化して明示的に渡す。target 文字列から feature suffix を捨てない。

release artifact の build では target または必要な codegen feature が未指定なら error とする。開発用 build で実機から補助的に検出する場合も、検出値を build log と artifact metadata に残し、配布 artifact へ暗黙に持ち込まない。artifact metadata には少なくとも次を記録する。

- canonicalized `ROCM_PATH`、ROCm release、compiler path と version
- `CMAKE_HIP_ARCHITECTURES`、`SLLM_HIP_CODEGEN_FEATURES`、code object ABI
- build profile と artifact format version
- link 対象の ROCm libraries と、確認可能な component version

### 実行時 ROCm library

build 時に選んだ ROCm tree だけでなく、process 起動時に dynamic loader が実際に解決した HIP/ROCr/ROCm libraries も互換性契約に含める。`LD_LIBRARY_PATH`、RPATH/RUNPATH、system cache により別 release をロードする可能性があるため、起動時に HIP runtime version と主要 library の解決済み absolute path を取得し、artifact metadata と照合する。

初期バージョンでは、build に使った ROCm release と実際にロードした ROCm user-space release が一致しない場合は起動 error とする。driver と user-space の互換範囲を将来許容する場合も、AMD の互換性資料と実機検証に基づく別 tuple として明示し、黙って警告だけで続行しない。診断には build 側と runtime 側の release、path、検出方法を含める。

## Compatibility tuple

互換性は Ubuntu、ROCm、GPU の独立した range ではなく、次の tuple を一単位として管理する。

```text
(Ubuntu release, point release, kernel, amdgpu driver,
 ROCm build release/root, resolved ROCm runtime release/library paths,
 GPU product, GPU target/architecture, codegen features, artifact metadata version)
```

`Ubuntu 24.04 対応`、`ROCm 7.14 対応`、`RDNA4 対応`という三つの記載から、その直積を対応済みと推論してはならない。GPU の対象範囲は [AMD GPU 互換性方針](amd-gpu.md) と対応づけるが、最終的な互換性状態は必ず具体的な tuple に付与する。point release、kernel、driver が複数許容される場合も、検証した値または AMD の互換性契約によって許容した集合を tuple record 内に明記する。

### Lifecycle の定義

software compatibility tuple の lifecycle は次の四つに統一する。

| Lifecycle | 定義 |
| --- | --- |
| `supported` | プロジェクトが互換性契約として受け入れ、不具合修正の対象とする tuple。原則として対応する実機検証 evidence を持つ |
| `experimental` | 実装中、試験的、または実機未検証の tuple。build 成功や vendor の公式掲載だけではここから昇格しない |
| `planned` | 対応する意図はあるが、実装・検証・修正が未完了であり、動作を保証しない tuple |
| `unsupported` | 対象外と決定した tuple、または既知の非互換性がある tuple。偶然動作しても互換性契約には含めない |

実機検証は lifecycle 値ではなく evidence である。検証した完全な tuple、日時、結果、対象機能を履歴として残し、その evidence を根拠に lifecycle を `supported` へ変更できる。逆に既知の不具合により `supported` から `unsupported` へ変更しても、以前の検証記録は消さない。

[GPU 互換性方針](gpu.md) の evidence 値 `vendor-supported`、`project-verified`、`unverified` は、vendor 公式掲載または sLLM 実機検証の根拠を表し、この lifecycle 軸とは役割が異なる。GPU evidence が十分でも OS、runtime library、artifact 条件まで一致しなければ software tuple は `supported` にならない。また、software lifecycle が `experimental` であることから vendor 公式対応の有無を推論してはならない。tuple record は lifecycle と GPU evidence を別 field に保持する。

### 初期候補 tuple

| Lifecycle | Ubuntu | Kernel | ROCm | GPU と artifact 条件 | 備考 |
| --- | --- | --- | --- | --- | --- |
| `experimental` | 24.04.4 LTS | GA 6.8 | build/runtime とも 7.14.0 | GPU、target、features ごとに個別 tuple | 主開発候補。現時点では sLLM 実機検証結果なし |
| `experimental` | Hot Aisle Ubuntu 24.04 | `6.8.0-124-generic`、amdgpu `6.16.13` | canonical user-space 7.14.0、HIP `7.14.60850`、hipBLASLt 1.4 | MI300X VF x1、`gfx942:sramecc+:xnack-`、exact `gfx942` artifact | Phase 12 `project-verified` scope。SR-IOV VF、single GPU、実行済みoperator/model/service/performanceに限定 |
| `planned` | 26.04 LTS | GA 7.0 | 7.14.0 | GPU、target、features ごとに個別 tuple | 将来検証候補 |

- Ubuntu 24.04.4 LTS、GA kernel 6.8、ROCm 7.14.0 の組み合わせを主系統候補とする。具体的な driver、GPU、target/features、dynamic library path まで確定した tuple だけを evidence の対象にする。
- Hot Aisle MI300X VMのPhase 12実測tupleはUbuntu 24.04、kernel `6.8.0-124-generic`、amdgpu `6.16.13`、
  MI300X VF、NPS1/SPX、VMM=trueである。provider driverを交換せず、provider imageの別ROCmをproduction pathに
  混ぜず、canonical `/opt/rocm/core-7.14`からHIP/hipBLAS/hipBLASLtを解決した。温度・電力は`amd-smi metric`の
  provider例外により取得不能で、0ではなく`unavailable`である。このtupleのevidenceを別OS/kernel/driver、
  bare metal、MI300A/MI325X、multi-GPUへ移植しない。
- Phase 10 local FP8 providerはROCm 7.14.0 / hipBLASLt 1.4.1を使用する。exact `gfx1201`はOCP E4M3FN
  native、exact `gfx1030`はemulationまたはload時BF16 conversionであり、別ROCm release、別target、
  CDNA3 FNUZの互換性を証明しない。
- Phase 11のROCm 7.14.0 local buildはexact `gfx942`、Code Object V6、wave64、`xnack=off`、`sramecc=on`を
  compile/linkした。Phase 12では上記Hot Aisle tupleでhipBLASLt FNUZ、wave64 operator、4B/9B BF16/FP8、
  contiguous-resident KV、OpenAI service、fixed llama.cpp比較を実機PASSし、そのscopeだけを
  `project-verified`とする。lifecycleは広い互換性保証を避けるため`experimental`のまま維持する。
- Phase 15のROCm 7.14.0 local runtimeではexact `gfx1030`/`gfx1201`、Code Object V6、wave32のtarget別binaryで
  weight NVFP4 packed-dequantを実行した。Qwen full-model、CLI/OpenAI service、cleanupの証拠はこのtupleに限定する。
  closeout再計測時にR9700だけ既存binaryを含むkernel imageがdriverから拒否されたため、その試行はPASSに含めず、
  Phase中に取得済みのR9700証拠とV620の最終再実行を分けて記録する。後続のtarget別performance binaryではR9700の
  short-odd/32-32 BF16/NVFP4 4 rowが再びPASSしたが、先行失敗試行を遡って成功扱いにはしない。
- Phase 15Oは同じlocal ROCm 7.14.0 tupleでexact `gfx1030`/`gfx1201`のNVFP4 decode/prefill providerと、exact
  `gfx1201`のFP8 dynamic量子化を実機検証した。target別release build、operator、full model、OpenAI service、cleanupを
  PASSした範囲だけを`project-verified`とし、別ROCm、別SKU、未実行のexact `gfx942`へ一般化しない。
- Phase 15Qも同じROCm 7.14.0、Code Object V6、wave32のtarget別release buildを使い、R9700/V620でGemma 4 12B-it
  BF16/S0/U0/O0のfull logits 96位置とlayer subsetを実行した。Unsloth artifactの取得・独立decodeはhost上で行い、model codeや
  Python runtimeをproduction GPU processへloadしていない。この証拠をROCm release、GPU SKU、W4A4/FP8 KVへ一般化しない。
- Phase 16も同じlocal ROCm 7.14.0 tupleで、exact `gfx1030`/`gfx1201`のtarget別binaryからFP8/NVFP4 KV appendと
  packed attentionを各17 case実行した。独立oracle、fallback false、cleanup 0、KV=8193のphysical commitmentをPASSした
  範囲だけを`project-verified`とする。exact `gfx942`はcompile/link-onlyであり、別tupleやCDNA3のruntime evidenceではない。
- Ubuntu 26.04 LTS と ROCm 7.14.0 の組み合わせは将来検証する `planned` tuple とする。AMD が ROCm 7.14.0 で Ubuntu 26.04 を掲載していても、sLLM による実機検証なしに Ubuntu 24.04 の結果を移植しない。
- 表にない Ubuntu、ROCm release、GPU の組み合わせは暗黙の `supported` としない。調査前は未分類であり、採用候補なら具体的な tuple を `planned` として追加する。

### 2026-08-03 local model-free evidence

次の実績は、同一immutable candidateのformal G0/G1で確認した限定的なevidenceである。初期候補のGA kernel 6.8とは異なるHWE kernel 6.17を使い、capability profile、resource gate、semantic数値kernel、model、性能と長時間安定性は未検証であるため、lifecycleは`experimental`のままとする。

| 項目 | 検証値 |
| --- | --- |
| lifecycle / evidence | `experimental` / `project-verified`（formal model-free G0/G1の範囲だけ） |
| candidate identity | commit `f393d688a051d2b73c8773d8a930a711592609bc` / tree `2ccda6e7c0614d585f26babc6b7c68ca51220bbe` |
| OS / kernel | Ubuntu 24.04.4 LTS / `6.17.0-35-generic` |
| amdgpu | `6.16.13` |
| ROCm build/runtime | system packages `amdrocm-core-sdk7.14-gfx1030`、`amdrocm-core-sdk7.14-gfx1201`（ともに `7.14.0-3`）、`https://repo.amd.com/rocm/packages-multi-arch/ubuntu2404` の `stable main`。root `/opt/rocm/core-7.14` |
| compiler / runtime | AMD clang 23.0.0git / HIP runtime `71460850` |
| package migration | legacy ROCm user-space packages、旧installation root、旧ROCm APT sourceを除去。amdgpu driver packagesは変更せず保持 |
| canonical GPU | V620 exact `gfx1030`、BDF `0000:03:00.0`、UUID `GPU-76a08c022586fed6`; R9700 exact `gfx1201`、BDF `0000:07:00.0`、UUID `GPU-a8e9ddefa2d60f55` |
| artifact | target別専用binary、Code Object V6、wave32、`xnack`/`sramecc=unsupported`; SHA-256 `gfx1030=40a55e8028355dd1b27b26886ccfef6d0b4085569d2656f90e7ebdc2be1a852c`、`gfx1201=69207b19c1146f73258db848fd5da74a25dd0a8e980b090ee09037da0dd2b1f5` |
| runtime libraries | `/opt/rocm/core-7.14/lib/libamdhip64.so.7.14.60850-0000000`、`/opt/rocm/core-7.14/lib/libhsa-runtime64.so.1.21.0` |
| G0/G1 scope | G0 identity/read-only health/process、G1の1、3、17、255、256、257 byte。各case 2 allocation、2 transfer、1 diagnostic dispatch、byte exact、CPU fallback/model/semantic opなし |

この結果はsemantic数値kernel、full model、性能、generic code object、複数GPU実行、長時間安定性、別GPU/SKU、またはvendor-supported OS/GPU tupleを証明しない。G0/G1 evidenceは検証したcandidate・artifact・canonical 2 GPUへだけ結び付ける。

### Phase 5 direct-runner の実行時 evidence

Phase 5 の direct runner は、UUID-primary の AMD-SMI `list -e` mapping を起点に、BDF、製品名、exact `gfx` target、HIP physical index を相互照合する。現在の検証値は V620 が `0000:03:00.0` / HIP index `1`、R9700 が `0000:07:00.0` / HIP index `2` である。AMD-SMI 26.5.0 は HIP の `GPU-*` UUID を `-g` selector として受け付けないため、個別 metric/process/static 読み出しは UUID mapping から解決した BDF に限定する。実行子processの `/proc/<pid>/maps` から `/opt/rocm/core-7.14` 配下の `libamdhip64.so` と `libhsa-runtime64.so` の実解決pathを取得し、ROCm 7.14.0 root外を拒否する。

各rowの pre/during/post evidence は温度、dynamic clock の観測範囲、power、performance level、profile、limits、runtime VRAM、AMD-SMI `monitor -v` の補助VRAM、ECC、process ownership、process-group cleanup、loader path digestを含む。dynamic clock の min/max、socket power telemetry、legacy aggregate `throttle_status` は観測値として記録する。R9700では300 W cap・330 W公開maximumを変更していないrunでも最大362 Wが観測され、倍率を後付けした瞬時telemetry gateでは有効なrunを安定に判定できなかった。power値単独はhard gateにせず、AMD-SMIが公開する明示的violation、ECC、slowdown温度以上、profile/limit/performance-level drift、foreign process、loader rootまたはlibrary digest違反をfail-closedとする。`static --clock`のcurrent levelと報告frequency levelは動的に変化し得るため、完全payloadを保存するがexact identity比較からは除外する。ROCm libraryはGPU処理開始後に固定ROCm root内の追加componentを遅延mapできるため、各観測のroot・path・content digestを独立検証し、検証済みpath集合の追加だけをdriftとしない。AMD-SMIのdynamic sensorが単発で`N/A`を返した場合だけ100 ms間隔で最大3回取得し、連続欠落はfail-closedとする。identity、process、ECC意味、明示violationはこのretry対象にしない。

このhostの V620/R9700 では AMD-SMI の violation/throttle accumulator fields が全て `N/A` であり、CLI help も violation reporting を MI300 以降に限定している。さらにR9700では2026-08-12の無負荷連続10回でlegacy aggregate `throttle_status`が8〜16 W、hotspot 42〜43℃でも`UNTHROTTLED` 4回、`THROTTLED` 6回と交互に現れ、Phase 3でも34℃・processなしの同現象を記録済みである。このfield単独はreasonを特定できるhard gateにせず、`accumulator_available=false` と制約を記録しつつ、ECC、温度、power、profile、limit、process、VRAM、loader evidenceを必須とする。MI300等でviolation fieldsが公開される場合はactive violationを成功runへ混ぜない。

明示allowlistされた外部processは、各観測時点でVRAM 1 MiB以下、GTT 16 MiB以下、GFX activity 0等のinert contractを独立に満たす場合だけ許可する。pre/post間でinert contextが消失またはlazy生成されること自体はbenchmark GPUへの干渉ではないため、存在一致を要求しない。各時点で存在するrecordのPID許可、resource上限、activityは引き続きfail-closedに検証する。

### 2026-08-13 Phase 6 A0 HIP VMM evidence

Ubuntu 24.04.4、kernel `6.17.0-35-generic`、amdgpu `6.16.13`、ROCm build/runtime 7.14.0の
local tupleで、canonical V620 `gfx1030`とR9700 `gfx1201`にtarget別standalone HIP binaryを実行した。
両targetでVMM minimum 4 KiB、recommended 2 MiB、reserve/create/map/access、contiguous-pointer kernel、
unmap/remap、event完了後cleanup、CPU byte oracleをPASSした。Qwen3.5-4B相当の128 MiB logical VA reserveは
physical delta 0 byteで、最初の2 MiB x 16 regionだけをcommitした32 MiBはcleanup後に復元した。

このevidenceはPhase 6 A0のmodel-free draft範囲だけを`project-verified`とする。software lifecycleは
`experimental`のままであり、production vAttention backend、full model、長時間安定性、別tuple、
Paged Attentionとの採用判断へは拡張しない。詳細identityとlatencyは
[Phase 6 history](../history/2026/08/11-20/openai-chat-completions-v1.md)を正とする。

### 2026-08-13 Phase 6 A1 virtual-contiguous KV evidence

同じlocal tupleとcanonical V620 `gfx1030` / R9700 `gfx1201`で、target別のFA2-style comparison probeと
actual public runtime production probeを実行した。comparisonはcontiguous、HIP VMM virtual-contiguous、
paged block-table accessorの36 caseを同一数値contractで比較し、production probeはtoken-major FP16 KVの
1023/1024/1025境界、physical commitment、未map拒否、cleanupを確認した。両targetともfallbackなし、
NumPyまたは独立BF16→FP16 oracle、pre/post ECC 0、process残留なしでPASSした。local aggregate SHA-256は
`453756b16f55ef81ff28dcb48cdebe69b9bdd83381b3a04202f94855af236021`である。

初期KV方式はこのtupleに限定したHIP VMM virtual-contiguous方式（vAttention型）とするが、software lifecycleは
`experimental`のままとする。比較kernelはupstream FlashAttention-2/CKではなくFA2-style proxyであり、
FA3/4 AMD動作、full model、service、長時間安定性、別tupleを証明しない。詳細は
[KV memory decision](../architecture/kv-memory.md)を正とする。

### 2026-08-14 Phase 6 A6 Qwen3.5-4B API service evidence

同じlocal tupleで、target別release binaryとQwen3.5-4B verified lockを使い、canonical V620 `gfx1030`と
R9700 `gfx1201`のOpenAI-compatible serviceを実行した。対象GPUはstable UUIDで一台だけ可視化し、serverは
論理device 0を使用した。raw HTTP non-stream/SSE、OpenAI Python client 2.44.0、stop、HIP dispatch後の
disconnect/recovery、1023/1024/1025 logical capacityを両targetでPASSした。completed requestはHIP-only、
fallbackなしであり、2 MiB page、32 MiB committed K/V、request/workspace allocation zero、GPU process
pre/post zero、ECC/health正常を記録した。

V620 report SHA-256は`b8ad41a3f35c693b98fc6629e5997413726fb8e9ad8dc16de21a49c20a874d8f`、
R9700 reportは`0648e41bb3a92ac60b82223a15b8ef2540ec9db7354da0ba29ecb5bf8c1f845f`である。
software lifecycleは`experimental`のままとし、複数GPU可視process、global physical indexでのworker選択、
別tuple、multi-GPU serving、長時間安定性は証明しない。詳細は
[Phase 6 history](../history/2026/08/11-20/openai-chat-completions-v1.md)を正とする。

### 2026-08-14 Phase 7 lifecycle evidence

daily/weekly/releaseのprofileはcanonical V620 `gfx1030` / R9700 `gfx1201`をこの文書と同じ
Ubuntu 24.04.4、kernel `6.17.0-35-generic`、amdgpu `6.16.13`、ROCm 7.14.0、exact UUID/BDFへ
固定する。GIMPS終了後の運用変更によりdailyもcanonical 2 GPUを観測し、全profileでforeign workloadを
性能PASSへ混ぜない既存Phase 5 preflightを再利用する。

local dirty candidateのdaily smokeでR9700 `gfx1201`、Qwen3.5-4B short-oddをwarmup 3回・計測10回
実行し、fallbackなし、health/cleanup PASSを確認した。中央値はTTFT 2.862 s、prefill
5.951 token/s、decode 1.672 token/s、E2E 12.440 s、resident/peak VRAM 8.412/8.541 GBだった。
GIMPS終了後のprofile revision 2ではcanonical 2 GPUを2/2 PASSし、V620中央値はTTFT 7.574 s、
prefill 2.246 token/s、decode 0.863 token/s、E2E 26.110 s、resident/peak VRAM 8.412/8.541 GBだった。
これは観測pathの再現性であり、immutable release evidence、性能優位性、長時間安定性を証明しない。
software lifecycleは`experimental`のままとする。

### 2026-08-14 Phase 8 BF16 optimized-path evidence

同じlocal tupleとcanonical V620 `gfx1030` / R9700 `gfx1201`で、Qwen3.5-4B BF16の
single-request optimized pathを実行した。target別binaryはROCm 7.14.0 rootだけをloadし、Matmulの
BF16 input/weight・FP32 accumulation・BF16 output、vAttentionのvirtual-contiguous FP16 K/V上の
FA2-style online softmax、exact target、fallbackなし、ECC 0、process cleanupを確認した。Matmulの
実model decode形状M=1,K=2560,N=9216はV620でcustom wave reduction、R9700でcontext-lifetimeの
`hipblasGemmEx`を選択する。後者もcheckpoint weightの転置・複製とlibrary workspaceを必要としない。

4B short-odd 17/17の3 warmup + 10 measured中央値は、V620がTTFT 1.099 s、prefill
15.555 token/s、decode 1.876 token/s、E2E 9.653 s、R9700がTTFT 0.683 s、prefill
25.102 token/s、decode 1.951 token/s、E2E 8.891 sだった。resident VRAMは両targetとも
8,411,592,192 bytes、peakは8,540,569,292 bytesである。2B/9B spot checkとOpenAI-compatible
non-stream/SSEも同じtupleでHIP-only、fallbackなし、cleanup 0をPASSした。これは該当local tupleの
project evidenceであり、別shape/model、multi-request、別ROCm/kernel tuple、長時間安定性へ一般化しない。
software lifecycleは`experimental`のままとし、詳細identityと全caseは
[Phase 8 history](../history/2026/08/11-20/phase8-bf16-optimization.md)を正とする。

同じtarget別binaryのcanonical O2はminimum、short-odd、255/256/257、prefill-long、decode-longの
7 caseを各GPUで3 warmup + 10 measured実行し、14/14 reportをPASSした。prefill-long中央値はV620が
94.734 prefill / 1.762 decode token/s、R9700が124.612 / 1.923 token/s、decode-longはV620が
30.389 / 1.867 token/s、R9700が45.507 / 1.955 token/sだった。pre/post ECC 0、GPU process 0、
VRAM baseline復帰を確認した。float64 oracle修正版のMatmul 17 caseとattention 16 caseも両targetで
再PASSし、fixtureのFP32 overflowとrow stride誤りをkernel defectと誤分類しないよう修正した。

Phase 8のproduction attentionはFA2-styleだけである。RDNA4 `gfx1200`/`gfx1201`向けの
FlashAttention-3-like pathは将来のtarget-specific比較課題であり、このevidenceやPhase 8完了条件には
含めない。

### 2026-08-14 Phase 9 engine-structure evidence

同じUbuntu 24.04.4、kernel `6.17.0-35-generic`、amdgpu `6.16.13`、ROCm build/runtime 7.14.0 tupleで、
canonical V620 `gfx1030`とR9700 `gfx1201`のtarget別release binaryを実行した。HIP Graph PoC、Matmul
17 case、4B short-odd/32x32、2B V620、9B R9700をHIP-only、fallbackなし、cleanup 0でPASSした。
R9700 optimized serverのraw OpenAI non-stream/SSEもPASSし、shutdown時のmodel/request/workspace current
bytesは0だった。

4B short-odd中央値はV620がTTFT/E2E 0.306/0.855秒、prefill/decode 56.91/29.69 tok/s、R9700が
0.051/0.490秒、377.46/37.20 tok/sである。これは該当dirty integration candidateとlocal tupleの限定evidenceで、
immutable release identity、GA kernel 6.8、別ROCm root、別GPU/SKU、multi-request、長時間安定性を証明しない。
software lifecycleは`experimental`のままとする。詳細は
[Phase 9 history](../history/2026/08/11-20/phase9-engine-structural-optimization.md)を正とする。

### 2026-08-16 Phase 16F low-bit tuple

Ubuntu 24.04.4、kernel `6.17.0-35-generic`、amdgpu `6.16.13`、ROCm build/runtime 7.14.0、LLVM 23の
local tupleで、exact `gfx1030`/`gfx1201`のW4A4、static FP8 KV、Unsloth Gemma 4 12B full mixed graph、通常CLIと
OpenAI non-stream/SSEを実行した。全実行はHIP-only、fallbackなし、shutdown cleanup 0だった。software lifecycleは
`experimental`のままで、別runtime/driver/kernelへ一般化しない。

### 2026-08-16 Phase 17 MTP/vision tuple

Phase 16Fと同じlocal software tupleで、exact V620 `gfx1030`とR9700 `gfx1201`のQwen3.5-4B MTP、vision、
multimodal text prefill/decodeを実行した。vision projected digestはtarget別浮動小数演算により異なるが、各target内のreplayは一致し、
全dispatch HIP、fallbackなし、cleanup 0だった。R9700ではCLI local PNGから1 token生成もPASSした。MTP逐次verifyは性能採用せず、
通常serviceはtarget-onlyを維持する。software lifecycleは`experimental`のままで、別tupleやlow-bit visionへ一般化しない。

### 2026-08-31 Phase 55 Gemma 4 MoE tuple

Ubuntu 24.04.4、kernel `6.17.0-35-generic`、amdgpu `6.16.13`、ROCm build/runtime 7.14.0、LLVM 23の
local tupleで、固定Gemma 4 26B-A4B NVFP4 artifactをexact `gfx1201`／`gfx1030`へ全resident化した。最終primary
`gfx1201` candidateはstatic E4M3 full/sliding KV、NVFP4 routed expert、CLI/server共通prepared executionをHIP-only、fallback
なし、nonfiniteなし、cleanup 0で実行した。sourceとcanonical GGUFの35-token digestは
`57c2f914705c86657a3537810e6ed5ba17972b67857c183135d1d0b8a117ccb1`へ一致し、通常CLIとdynamic API/WebUI serviceも同じtupleで
load／生成／cancel／unload／clean shutdownをPASSした。`gfx1030` full-resident値はstate API前のdraft evidenceであり、最終sourceでは両targetの
operator matrixを保持する。このtuple以外のdriver/runtime、GPU、multi-GPU、長時間安定性を主張しない。

### 2026-08-31 Phase 57 DeepSeek V4 route-operator tuple

Phase 55と同じUbuntu 24.04.4、kernel `6.17.0-35-generic`、amdgpu `6.16.13`、ROCm build/runtime 7.14.0、LLVM 23 tupleで、
DeepSeek V4専用top-6 MoE routing operatorをexact `gfx1030`／`gfx1201`へtarget別Code Object V6／wave32 binaryとして実行した。
`gfx1030` binary SHA-256は`a3190e16ce3abceb304c19fece79662070ecd1d01bc626eed2a8f5f373e162c2`、`gfx1201`は
`ba7c0f2ef30c3e3668e97acd2a82984f07dc79acad90e4882b570a80eb629a48`である。M=1/3/5/17のscore／hash routeと不正入力を
独立oracleへ照合し、不正入力の公開completion fail-close、HIP-only、fallback 0、cleanup 0をPASSした。このtupleはmodel-free operatorだけを対象とし、official checkpointの
resident、full graph／service、性能、multi-GPU、別driver/runtime/GPUへ一般化しない。

### 2026-08-31 Phase 58 MiniMax M3 route-operator tuple

Phase 57と同じlocal software tupleで、MiniMax M3専用sigmoid top-4 MoE routing operatorをexact `gfx1030`／`gfx1201`へ
target別Code Object V6／wave32 binaryとして実行した。`gfx1030` binary SHA-256は
`b14988e6916286c730720a49b997ec99fed052d5c8f0fba4cda916f619247edc`、`gfx1201`は
`212bfcf6f9dd28d2773d01d0890edc9d6165566b8cdb064f55380fcddcd27bc7`である。M=1/3/5/17と不正入力を独立oracleへ照合し、
公開completion fail-close、HIP-only、fallback 0、KFD process残留なし、VRAM baseline復帰をPASSした。このtupleは
model-free operatorだけを対象とし、official checkpointのresident、MSA／full graph／service、multimodal／MTP、性能、
multi-GPU、別driver/runtime/GPUへ一般化しない。

### 2026-08-31 Phase 59 DiffusionGemma host-foundation tuple

Phase 59はRust host testsとfixed-revision metadata／bounded safetensors header照合だけを実行した。公式BF16 shard file合計
51,647,701,024 bytesが単一32 GiB GPUへ収まらないため、新しいHIP binaryまたはexact `gfx1030`／`gfx1201` full-model runはない。
従ってsoftware／GPU compatibility表は拡張せず、identity／graph／sampler／GGUF write-disabled dry-runの証拠をGPU実行PASSへ
読み替えない。

### 2026-08-16 Phase 18 exact MTP tuple

Phase 17と同じUbuntu 24.04.4、kernel `6.17.0-35-generic`、amdgpu `6.16.13`、ROCm 7.14.0、LLVM 23 tupleで、
exact `gfx1030`/`gfx1201`のserial-equivalent target block、BF16/FP8 target、FP16/static-FP8 KVを実行した。raw target logitsと
accepted-prefix KVを逐次M=1へ照合し、CLI、OpenAI non-stream/SSE/cancel/recovery/shutdownをR9700でPASSした。R9700だけ
検証済みBF16 greedy rowを通常内部MTP providerへ採用し、V620と未計測tupleはtarget-onlyを維持する。software lifecycleは
`experimental`のままで、別runtime/driver/kernelへ一般化しない。

### 2026-08-16 Phase 19 Qwen3.5 MoE tuple

Phase 18と同じUbuntu 24.04.4、kernel `6.17.0-35-generic`、amdgpu `6.16.13`、ROCm 7.14.0、LLVM 23、
Code Object V6、wave32 tupleで、exact V620 `gfx1030`とR9700 `gfx1201`のQwen3.5-35B-A3B MXFP4 sparse MoEを
target別release buildで実行した。actual-weight oracle、40層full-model prefill/decode、CLI/OpenAI service、cancel/recovery、
seeded sampling、shutdownは全dispatch HIP、fallbackなし、cleanup 0をPASSした。software lifecycleは`experimental`のままで、
別runtime/driver/kernel、別artifact、multi-GPU、長時間運転へ一般化しない。

### 2026-08-17 Phase X llama.cpp Qwen3.8 quantized-KV tuple

Linux kernel `6.17.0-35-generic`、ROCm 7.14.0、HIP runtime 7.14.60850、LLVM 23、rocprofv3 1.3.2で、
llama.cpp build 901 commit `4df29be4f4c3673f428170fda944a5b19f743bb8`をexact V620 `gfx1030`と
R9700 `gfx1201`向けにbuildした。Qwen3.8-27B Q5_K_XL、context 262,144、Q5_1 model/draft KV、MTP幅3の
9,435-token実code promptを使用した。

baselineの`GGML_CUDA_FA_ALL_QUANTS=OFF`ではHIP prefill/decodeがV620 60.99/6.81、R9700
69.50/12.50 tok/sだった。`GGML_CUDA_FA_ALL_QUANTS=ON`の1 warmup + 5 measured中央値はV620
340.80/33.42、R9700 779.06/41.93 tok/sで、Q5_1 Flash Attention exact Qwen shapeをCPU numerical oracleへ
照合して両target各18/18 PASSした。CPU/backend fallbackとGTT spillはない。これは外部llama.cpp local-subagent runtimeの
限定evidenceであり、sLLM software lifecycle、別model/KV、別tuple、multi-GPU、長時間安定性を証明しない。詳細は
[Phase X bounded summary](../../ci/matrix/phase-x-qwen38-amd-summary-v1.json)を正とする。

同日の非運用ベンチマークでは、2基のV620をllama.cppのexperimental tensor split `1,1`で使用し、actual context
1,048,576、Q5_1 target/draft KV、MTP幅3、batch/ubatch 512/128を起動した。9,435-token code promptの
1 warmup + 3 measured中央値はprefill/decode 416.80/47.90 tok/s、合計observed peak VRAMは
66,560,937,984/68,685,922,304 byte（96.91%）、GTTは40,599,552 byteだった。2-hop PCIe構成でinternal AllReduceは
初期化できずmeta-backend butterflyを使い、token samplerはCPUへfallbackした。1M-token実入力は未実施であり、
maximum-context correctness、長時間安定性、strict scalingを証明しない。後続のユーザー決定で同じV620×2 tensor shapeを
491,520 context/slot、983,040 total、parallel 2、non-unified KVへ縮小して通常起動へ昇格した。
[TP2 1M bounded summary](../../ci/matrix/phase-x-qwen38-v620-tp2-1m-summary-v1.json)は昇格前の計測値を保持する。

同じbuild 901に`gfx1030;gfx1201`を含む比較用HIP buildを追加し、独立V620×2、V620×2 layer/tensor、
R9700+V620×2 layer/tensorを11,058-token code promptの同時2要求で比較した。V620だけの最大throughputは
独立2 server、V620だけで524,288 context/slotを必要とする場合はexperimental tensor、3基を明示的に空けられる
場合のsingle-process候補はlayer split `5,2,2`となった。rowはupstream非推奨のため除外した。比較buildと全一時serverは
停止した。後続決定ではR9700を占有しないV620×2 tensorの縮小構成だけを通常runtimeへ昇格した。詳細は
[multi-GPU selection summary](../../ci/matrix/phase-x-qwen38-multi-gpu-selection-v1.json)を参照する。

現行local subagentのagent layerはPi coding agent 0.84.2をChat Completions経由で使用する。model catalogはcombined totalではなく
actual 491,520 context/slot、max output 8,192へ同期し、read-only/workspace-write Landlock sandboxと2 process leaseを使う。
2つの同時taskは別slotで完了し、3つ目はqueueせずstatus 75で終了する。DeepSeek HarnessはResponses translation問題の
compatibility/debug経路に限定する。このagent-layer変更はllama.cpp build、model、GPU/KV/MTP tupleを変更しない。
運用と実測は[Local Qwen3.8 subagent](../development/local-qwen-subagent.md)を正とする。

### 2026-08-19 Phase 30 RDNA4 attention/KV tuple

Ubuntu 24.04.4、kernel `6.17.0-35-generic`、amdgpu `6.16.13`、ROCm build/runtime 7.14.0、LLVM 23、
Code Object V6、wave32 tupleで、canonical R9700 exact `gfx1201`とV620 exact `gfx1030`をtarget別buildした。
gfx1201ではnative E4M3FN readの`v_cvt_f32_fp8`とwave shuffle providerをactual dispatchし、gfx1030では
software decode/scalar baselineを維持した。両targetのFP16/FP8各17 case、gfx1201全256-code probe、
Qwen3.5-4B BF16 full modelをHIP-only、fallbackなし、cleanup 0で実行した。software lifecycleは`experimental`のままで、
別ROCm/driver/kernel、別SKU、matrix attention、長時間運転へ一般化しない。

### 2026-08-19 Phase 31 chunked prefill tuple

Phase 30と同じUbuntu 24.04.4、kernel `6.17.0-35-generic`、amdgpu `6.16.13`、ROCm build/runtime 7.14.0、
LLVM 23、Code Object V6、wave32 tupleで、canonical R9700 exact `gfx1201`とV620 exact `gfx1030`をtarget別buildした。
Qwen3.5-4B BF16 GGUFの10,001-token FP16/dynamic FP8を両target、16,385-token 2-chunk FP16/dynamic FP8を
gfx1201でHIP-only、fallbackなし、cleanup 0として実行し、gfx1201 OpenAI non-stream/SSEも10k+ promptでPASSした。
software lifecycleは`experimental`のままで、別runtime/driver/kernel、別artifact、品質、長時間運転へ一般化しない。

### 2026-08-19 Phase 32 native FP8 append tuple

Phase 31と同じOS/kernel/driver、ROCm 7.14.0、LLVM 23、Code Object V6、wave32 tupleでexact gfx1201/gfx1030を
target別buildした。gfx1201 FP8 append code objectだけが`v_cvt_pk_fp8_f32`を含み、gfx1030はsoftware encodeを維持した。
dynamic/static FP8 production oracle 68/68 caseと、Qwen3.5-4B BF16 GGUFのgfx1201 10,001/16,385、gfx1030
10,001 inputをHIP-only、fallbackなし、cleanup 0で実行した。これは同じtupleの限定証拠であり、別ROCm/compiler、
gfx1200、別RDNA4 SKUへnative instruction availabilityまたは性能を一般化しない。

### 2026-08-20 Phase 33 Full Attention tuple

Phase 32と同じOS/kernel/driver、ROCm 7.14.0、LLVM 23、Code Object V6、wave32 tupleでcanonical exact
gfx1201/gfx1030をtarget別release buildした。FP16/dynamic FP8/static FP8/NVFP4 × 2 target × 29 caseの
232/232 oracle、R9700 FP16 10,000-prompt、V620 FP16 4,108-prompt、R9700 dynamic FP8 OpenAI lifecycle、
wrong-target拒否を最終identityでPASSした。採用scopeはC1 `M=1`/KV>=1,024とC2 `M>=64`/GQA4/head dim 256に
限定し、software lifecycleは`experimental`のままである。別ROCm/compiler、別SKU/head shape/model、matrix provider、
長時間運転へ一般化しない。[Phase 33 summary](../../ci/matrix/phase33-full-attention-summary-v1.json)を正とする。

### 2026-08-20 Phase 34 V620 long-prefill BF16 matmul tuple

Phase 33と同じUbuntu 24.04.4、kernel `6.17.0-35-generic`、amdgpu `6.16.13`、ROCm build/runtime 7.14.0、
LLVM 23、Code Object V6 tupleでcanonical exact gfx1030/gfx1201をtarget別release buildした。gfx1030の限定6 shapeに
existing hipBLAS routeを追加し、10,001-token FP16 KV full model、両targetの18-case matmul oracle、gfx942 compile-only、
wrong-target拒否をPASSした。software lifecycleは`experimental`のままで、別ROCm/compiler/driver、別SKU/model shape、
library solutionの安定性へ一般化しない。[Phase 34 summary](../../ci/matrix/phase34-v620-prefill-matmul-summary-v1.json)を正とする。

### 2026-08-20 Phase 35 long-context Full Attention/GDN tuple

Phase 34と同じUbuntu 24.04.4、kernel `6.17.0-35-generic`、amdgpu `6.16.13`、ROCm build/runtime 7.14.0、
LLVM 23、Code Object V6、wave32 tupleでcanonical exact gfx1030/gfx1201をtarget別release buildした。
Full Attention 232 case、GDN 12 case、10,001-input combined full model、wrong-target拒否をfinal sourceでPASSし、
gfx942 wave64はcompile-onlyを通した。software lifecycleは`experimental`のままで、別ROCm/compiler/driver、別SKU/model shape、
長時間運転、gfx942実行へ一般化しない。[Phase 35 summary](../../ci/matrix/phase35-attention-gdn-summary-v1.json)を正とする。

### 2026-08-21 Phase36 Session A Hot Aisle MI300X tuple

Session A の software compatibility は Hot Aisle MI300X VF x1 の exact tuple に限定する: Ubuntu 24.04.4、kernel
`6.8.0-124`、amdgpu `6.16.13`、ROCm 7.14.0、HIP `7.14.60850`、LLVM 23、`gfx942:sramecc+:xnack-`、wave64、
304 CU、HBM `205,822,885,888` bytes、NPS1/SPX、VMM `true`。実行中はROCm 7.14 rootをlogical `/opt/rocm`へ限定し、
終了後はprovider `/opt/rocm-7.2.4`へ復元した。release artifact はlogical `gfx942`、唯一のdevice bundle
`gfx942:sramecc+:xnack-`、Code Object V6、ELF flags `0xE4C`、全kernel wave64で、generic/別target artifactを含めない。
最終artifactのoperator matrixは99/99 PASSである。

Qwen3.5-4B BF16/FP8 GGUF の固定短生成は HIP-only、fallback `0`、cleanup `0`。gfx942 FP8 は OCP E4M3FN storage を
native FNUZ resident bytes/scales へ正しく rebasing/conversion し、raw reinterpret をしない。Hello token は BF16
`[11,353,2688,4313,310]`、FP8 `[11,353,1044,4313,310]`（cross-provider N1 差）で、Unicode/stop も同じ contract を満たした。
Phase29 GDN wave32 scope leak は修正済みで wave64 の sequential norm を維持する。
second resident request/model reuse、drop後allocation 0も確認した。post-run process `0`、全sysfs RAS block CE/UE `0`、VRAM
baseline復帰、provider `/opt/rocm` link復元、VM外raw退避を確認した。

この記録は Session A の `project-verified` scope のみであり、Sessions B-D、9B、low-bit KV/long context、MTP、vision、service、
performance、multi-GPU、別SKU/VM/bare-metal を含まない。software lifecycle は `experimental` のままとする。

### 2026-08-21 Phase36 Sessions B/C Hot Aisle MI300X tuple

Session Aと同じUbuntu 24.04.4、kernel `6.8.0-124`、amdgpu `6.16.13`、ROCm 7.14.0、HIP `7.14.60850`、
LLVM 23、exact `gfx942:sramecc+:xnack-`、wave64 tupleでSessions B/Cを実行した。Session Bは4 KV encodingの
Full Attention 116 case、FP16 KV state 19 case、独立low-bit oracle、およびBF16 targetのFP16/dynamic FP8 KVの
10,001 input / 2 output × 6 chunk設定をPASSした。全model rowはHIP-only、fallbackなし、cleanup 0で、HBM/GTTは
共通baselineへ復帰した。

Session CはBF16/FP8 targetの代表MTP、PNG/JPEG/WebP visionとlazy residency、OpenAI profile v1のraw/official-client
non-stream/SSE、reasoning、stop、seed、disconnect/recovery、二並行queue、graceful shutdownをPASSした。
`amd-smi metric`はprovider実装の`partition`例外で`unavailable`だったが、static identity、sysfs memory、process audit、
runtime reportを0へ置換せず別に保持した。GPU work後はROCm 7.14 bindを解除し、provider `/opt/rocm-7.2.4`へ復元した。
このtupleはSession D、9B、反復性能、llama.cpp/profile、別OS/driver/ROCm/SKUを含まず、software lifecycleは
`experimental`のままである。詳細は[Session B summary](../../ci/matrix/phase36-mi300x-session-b-summary-v1.json)と
[Session C summary](../../ci/matrix/phase36-mi300x-session-c-summary-v1.json)を正とする。

### 2026-08-21 Phase36 Session D Hot Aisle MI300X tuple

Sessions A〜Cと同じUbuntu 24.04.4、kernel `6.8.0-124`、amdgpu `6.16.13`、ROCm 7.14.0、HIP
`7.14.60850`、LLVM 23、exact `gfx942:sramecc+:xnack-`、wave64 tupleでSession Dを実行した。BF16/FNUZ FP8の
各5ケースを3 warmup＋10 measuredでPASSし、fixed llama.cpp `b10453`のE1比較とBF16 10,001/2 rocprofv3を取得した。
全rowは単一ROCm loader closure、HIP-only、fallbackなし、process終了後HBM/GTT baseline復帰を満たした。
当該tupleの4B反復性能/profileだけを`project-verified` evidenceへ追加し、9B、Gemma/MoE、長時間安定性、
別OS/driver/ROCm/SKUは含めない。Sessions A〜DをPhase 36の完了範囲としてcloseし、VMはユーザーが削除した。
software lifecycleは`experimental`のままで、[Session D summary](../../ci/matrix/phase36-mi300x-session-d-summary-v1.json)を正とする。

### 2026-08-21 local R9700 direct E2E tuple

Ubuntu 24.04.4、kernel `6.17.0-35-generic`、amdgpu `6.16.13`、ROCm 7.14.0、HIP `7.14.60850`、
LLVM 23、canonical R9700 exact `gfx1201`のtupleで、Qwen3.5-4B BF16、FP16 KV、10,001/2 direct rowを
3 warmup＋10 measured実行した。sLLM / fixed llama.cpp `b10453`のE2E中央値は
`3.936429665/2.063845785`秒、比`1.90733x`で、生成`[23066,23066]`、HIP-only/full-offload、cleanup 0、
終了後process 0を確認した。GGUF bytes/tensor setは異なるE1であり、このtuple以外の性能やstrict identityを主張しない。
software lifecycleは`experimental`のままで、[R9700 summary](../../ci/matrix/r9700-sllm-llama-e2e-v1.json)を正とする。

### 2026-08-22 Phase41 local state tuple

Ubuntu 24.04.4、kernel `6.17.0-35-generic`、ROCm 7.14.0、HIP `7.14.60850`、LLVM 23で、canonical V620
exact `gfx1030`とR9700 exact `gfx1201`のopaque KV/linear state fork、COW、encoding-native export/import matrixをPASSした。
FP16 63/64/65/127/128/129、dynamic/static FP8、NVFP4、linear active slot/scratchをstate oracleへ照合し、target-only、
fallbackなし、cleanup/process/ECC failure 0を確認した。このtupleはPhase 41 state contractに限定し、full-model性能、長時間運転、
別driver/ROCm/SKU、production checkpoint/contextの全組合せへ一般化しない。software lifecycleは`experimental`を維持する。

同じsourceはexact `gfx942:sramecc+:xnack-`、wave64でcompile/linkしたが、MI300X real runはVM再確保後へdeferredした。
[Phase41 GPU summary](../../ci/matrix/phase41-state-gpu-summary-v1.json)を正とする。

### 2026-08-24 Phase49 exact `gfx1030`限定性能tuple

Phase 49の性能candidateはcanonical V620 exact `gfx1030`（UUID `GPU-76a08c022586fed6`、BDF `0000:03:00.0`）、Ubuntu 24.04.4、kernel `6.17.0-35-generic`、
amdgpu `6.16.13`、ROCm build/runtime 7.14.0、HIP `7.14.60850`、LLVM 23、Code Object V6、wave32の
target専用tupleに限定した。GQA4 decodeのP32 partitionだけをKV長4,096以上、head dimension 256、FP16 KV、
`M=1`で既定採用し、long-prefill v2とHIP Graphは採用しなかった。これはV620の上記機能scopeだけを
`project-verified`とする記録であり、software lifecycleは`experimental`のまま維持する。R9700/MI300X、
別SKU、別driver/ROCm、generic artifactへselector・閾値・binaryを一般化しない。詳細は
[数値変更台帳](numerical-output-changes.md)と[Phase 49以降ロードマップ](../history/2026/08/21-31/phase37-plus-mi300x-and-llama-gap-roadmap.md)を正とする。

### 2026-08-24 Phase50 local R9700 / MI300X handoff tuple

Phase 50のR9700実機tupleは、Ubuntu 24.04.4、kernel `6.17.0-35-generic`、amdgpu `6.16.13`、ROCm 7.14.0、
HIP `7.14.60850`、LLVM 23、Code Object V6、wave32、canonical UUID `GPU-a8e9ddefa2d60f55`、BDF
`0000:07:00.0`、exact `gfx1201`である。build/runtimeのROCm rootとloader pathを閉じ、generic/multi-arch
artifactを性能evidenceへ混ぜない。

このtupleで検証した実機scopeは、固定Qwen3.5-4B BF16、FP16 KV、`M=1`、単一active requestにおけるResidual RMSNorm、
GDN projection、MLP gate/up/SiLU、GQA P32のA/B比較、最終通常行と長行の計測・未達判定である。R9700のこの機能scopeだけを
`project-verified`とし、lifecycleは`experimental`のままとする。100,000-token prefillはOOMで完走せず、成功scopeへ含めない。
20,000-token decodeの完走結果はPhase 50履歴とsummaryへ固定し、この方針文書では数値を重複しない。完了行、未達理由、candidate分類は
[Phase 50履歴](../history/2026/08/21-31/phase50-r9700-port-and-mi300x-handoff.md)と
[数値変更台帳](numerical-output-changes.md)を参照する。

MI300X側は論理target `gfx942`、feature付きdevice/codegen target `gfx942:sramecc+:xnack-`、Code Object V6、wave64、
`xnack=off`、`sramecc=on`のcompile/linkとhost selector非選択だけをPhase 50で検証する。production Cargoの
`CMAKE_HIP_ARCHITECTURES`はlogical `gfx942`を使い、feature suffix付きtargetはdirect CMake probeだけで扱う。このevidenceはMI300X実機runtime/PASSではなく、
Phase 51でfresh preflight・7行・wave64 providerを検証するためのhandoffである。既存Hot Aisle MI300Xの
別tuple runtime evidenceを更新せず、Phase 50のcompile/host結果を`project-verified`へ昇格しない。
R9700のscopeをRDNA4全体、MI300XのscopeをCDNA3全体、他OS・kernel・driver・ROCm・SKUへ推論せず、
全tupleのlifecycleは`experimental`を維持する。

### 2026-08-24 Phase52 local R9700 100k tuple

Phase 50と同じUbuntu 24.04.4、kernel `6.17.0-35-generic`、amdgpu `6.16.13`、ROCm 7.14.0、
HIP `7.14.60850`、LLVM 23、Code Object V6、wave32、UUID `GPU-a8e9ddefa2d60f55`、BDF `0000:07:00.0`、
exact `gfx1201`を使った。source commit `3ed002c476b49417cc702119e37c2389cefb96bc`からfresh buildした
release binary SHA256は`79b0099f0c8981c46d1629debaf2aacfe551107adb13ec00465f4ebce11c8f81`である。

固定Qwen3.5-4B BF16 GGUF、FP16 KV、単一request、自動prefillで`10,001/2`を13/13、`100,000/2`を4/4 PASSした。
100kはlogical capacity 131,072のresident KVと実効chunk 2,048を使用し、生成、HIP-only、fallback/cleanup 0、process消滅、
HBM/GTT baseline復帰を確認した。このsoftware tupleとcaseだけを追加実機evidenceとし、別OS/kernel/driver/ROCm/SKU、
batch/parallel、他modelへ一般化しない。software lifecycleは`experimental`のままで、
[Phase 52 summary](../../ci/matrix/phase52-r9700-kv-commit-summary-v1.json)を正とする。

### 2026-08-27 Phase53 local RDNA descriptor v1履歴とv2 follow-up

Phase 53の追加reportはcanonical local RDNAのexact `gfx1030`／`gfx1201`、binary、model、dataset、policy identityを固定するが、
OS／kernel／driver tuple全体はreport内で再固定していない。このため新しいsoftware tupleの検証とは扱わず、実機format／品質scopeを
exact targetへだけ限定する。exact target専用release binaryのSHA-256はgfx1030 quality
`513318543504c9d0e1a8fe4af43dcae5da7ffe7ca5ae2af4132a993fc5eb1754`、gfx1201 quality
`97de3a1711843aef2fc3e07473dc13fd40875067ecd485ace852d7705988914a`、format correctnessはgfx1030
`4e09989f6d2f3b38eeaa1b6aca70e4b81c5c57c08331e0c5fd4bb670faf04c66`、gfx1201
`40518586be413d2b370a12e2dba05ff2ba3661392853d64a224b48e38a294e3d`である。

このtupleでdescriptor v1 block16／standard OCP MXFP8のGPU correctnessはPASSしたが、freeze済みquality threshold未達により
gfx1030／gfx1201とも旧recipeを`retain-fp16`とした。performance/resourceはearly-stopで未実行、旧runtime mapping候補は空、
software lifecycleは`experimental`のままとする。このbinary／report evidenceはdescriptor v1履歴で、v2へ流用しない。
gfx942はfresh Phase 53 tuple evidenceがなく`insufficient-evidence`であり、
別OS、kernel、driver、ROCm、SKUへ推論しない。summaryと個別digestは
[Phase 53履歴](../history/2026/08/21-31/phase53-kv-fp8-block16-default-adoption.md)を正とする。
2026-08-27のユーザー指示によりMI300X実機は追加検証項目がまとまるまで延期し、Hot Aisleの固定IP疎通をVMまたはsoftware tupleの
存在証拠として扱わない。

2026-08-30のユーザー決定でblock16 production経路を廃止し、上記結果を履歴へ固定した。同じsoftware tupleで
reviewed Qwen3.5-4B BF16 dense text／full attention／single GPU／head dim 256の省略時KVはstandard OCP
`kv-mxfp8-e4-v1`となる。exact `gfx1030`、`gfx1201`、`gfx942:sramecc+:xnack-`でOCP E4M3FN／block 32／E8M0を使い、
明示`fp16`をrollbackとして残す。V620 `gfx1030`とR9700 `gfx1201`ではfresh direct GPU byte／attention oracleをPASSしたが、
一回のfull-model測定でgfx1201 top-1一致が`0.85`となりfreeze済み`>=0.99`に未達だった。default変更はユーザー明示決定に
基づくN2であり、software lifecycleまたはrelease品質の自動昇格ではない。gfx942はfresh実GPU evidence未取得である。

### 2026-08-31 Phase56 local R9700 Gemma 4 MTP tuple

Phase 50と同じlocal R9700、ROCm 7.14.0、HIP 7.14.60850、LLVM 23、Code Object V6、wave32、UUID
`GPU-a8e9ddefa2d60f55`、exact `gfx1201`でfresh release buildした。最終binary SHA-256は`sllm`
`37368b55f9abe886e2316342e2b7e40e2c6f5e2ad4b24be516f2059338b3a6b4`、`sllm-server`
`0c9a99dfc233ccb7a66110f035b076876065be3e24ee3e05091227a6827fd4a1`である。

reviewed mixed low-bit Gemma 4 12B targetとBF16 assistantをcontext 2,048、greedy width 1で実行し、CLI benchmark、static server、
source-tree WebUI dynamic server、metrics、cancel、unload、shutdownをPASSした。これはmodel／artifact／software tupleを同時に固定した狭いevidenceであり、
別ROCm、driver、kernel、GPU、Node-free release packagingへ一般化しない。

### 2026-08-31 Phase60 local RDNA Ministral 3 tuple

local Ubuntu 24.04.4、kernel `6.17.0-35-generic`、amdgpu `6.16.13`、ROCm 7.14.0、HIP `7.14.60850`、
LLVM 23のtupleで、公式Ministral 3 3B BF16 GGUFをexact `gfx1030`とexact `gfx1201`向けにそれぞれbuild／実行した。
両targetは同じ短いtoken列、HIP-only、fallback false、394／394 dispatch、shutdown cleanup 0となった。

固定llama.cpp oracleとは3番目の生成tokenから一致しないため、このtupleはbuild／dispatch／lifecycle到達だけを示す。
Ministral 3のproduction数値品質、別OS／driver／ROCm／GPU、または広いcontext／performance evidenceへ一般化しない。

## 公式資料

- [Ubuntu releases](https://releases.ubuntu.com/) — Ubuntu 24.04 LTS および 26.04 LTS の公式 release 情報
- [Announcing Rust 1.85.0 and Rust 2024](https://blog.rust-lang.org/2025/02/20/Rust-1.85.0/) — Rust 2024 の安定化
- [Cargo: Rust-version aware resolver](https://doc.rust-lang.org/stable/edition-guide/rust-2024/cargo-resolver.html) — resolver 3 と virtual workspace での明示設定
- [Announcing Rust 1.97.1](https://blog.rust-lang.org/2026/07/16/Rust-1.97.1/) — 開発 pin の release 情報
- [ROCm Core SDK 7.14.0 release notes](https://rocm.docs.amd.com/en/docs-7.14.0/about/release-notes.html) — OS 対応、component versions、LLVM 23
- [ROCm Core SDK components](https://rocm.docs.amd.com/en/docs-7.14.0/components/core.html) — ROCm 同梱 compiler と core components
- [AMD ROCm multi-architecture APT repository](https://repo.amd.com/rocm/packages-multi-arch/ubuntu2404) — 現在の Ubuntu 24.04 system package source
- [Install AMD ROCm 7.14.0](https://rocm.docs.amd.com/en/docs-7.14.0/install/rocm.html) — self-contained multi-architecture tarball と custom install directory の公式手順
- [ROCm environment variables](https://rocm.docs.amd.com/en/docs-7.14.0/reference/environment-variables/index.html) — Linux における `ROCM_PATH` と compiler path
- [CMake 3.21 release notes](https://cmake.org/cmake/help/v3.21/release/3.21.html) — 最小 CMake version の一次資料
