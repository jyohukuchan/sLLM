# ソフトウェア互換性方針

## 目的

この文書は、uLLM のビルドおよび実行に使う OS、ツールチェーン、ROCm の互換性契約を定義する。ここに記すバージョンは初期決定であり、実装や実機検証で問題が判明した場合は、コードだけを回避的に変更せず、この文書の tuple と判断理由も更新する。

## 基準ツールチェーン

| 項目 | 初期決定 | 方針 |
| --- | --- | --- |
| OS | Ubuntu 24.04 LTS | 主開発・主配布環境。point release と kernel も compatibility tuple に記録する |
| Rust edition | 2024 | workspace 全体で統一する |
| Rust MSRV | 1.85.0 | `rust-version = "1.85.0"` として公開クレートに明記する |
| Rust 開発 pin | 1.97.1 | 2026-08-02 時点の開発用 toolchain。`rust-toolchain.toml` で固定する |
| Cargo resolver | 3 | virtual workspace の `[workspace]` に `resolver = "3"` を明記する |
| Cargo lockfile | commit する | uLLM は application であるため、workspace root の `Cargo.lock` を version 管理する |
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

1. 環境変数 `ULLM_ROCM_PATH`
2. スクリプト実行前から定義されている環境変数 `ROCM_PATH`
3. 標準配置 `/opt/rocm/core-7.14`

`ULLM_ROCM_PATH` またはスクリプト実行前から定義されている `ROCM_PATH` が定義されて空の場合は明示的に失敗させる。選択した root が存在しない場合、または検査に失敗した場合も明示的に失敗させ、既定 rootや別 releaseへfallbackしない。

選択後は path を canonicalize し、compiler、HIP headers、CMake package、runtime、device libraries をすべてその root から解決する。HIP compiler は原則として安定した entry point `${ROCM_PATH}/bin/amdclang++` を使い、その symlink を解決した実体も同じ ROCm root 内にあることを検査する。package manager 配置と tarball 配置で LLVM の内部 directory が異なるため、`${ROCM_PATH}/llvm` または `${ROCM_PATH}/lib/llvm` を無条件に仮定しない。発見した root、ROCm release、`amdclang++ --version` の LLVM major を configure 時に検査し、ROCm 7.14.0、LLVM 23、または期待する配置と一致しない場合は明示的に失敗させる。暗黙に system `clang++`、別の `/opt/rocm-*`、別 release の library へフォールバックしてはならない。

「ROCm components は同一 release」とは、各 component 固有の内部バージョン番号を `7.14.0` に揃えるという意味ではない。ROCm 7.14.0 の配布物・repository として組み合わせて公開された compiler、HIP runtime、ROCr、math libraries、headers、device libraries を混在させずに使う、という意味である。

### GPU target と codegen feature

HIP binary の target は host の自動検出結果だけで決めず、Cargo から CMake へ `CMAKE_HIP_ARCHITECTURES` を明示的に渡す。`xnack`、`sramecc`、wavefront size など、binary compatibility または命令生成を変える codegen feature は project 固有の `ULLM_HIP_CODEGEN_FEATURES` に正規化して明示的に渡す。target 文字列から feature suffix を捨てない。

release artifact の build では target または必要な codegen feature が未指定なら error とする。開発用 build で実機から補助的に検出する場合も、検出値を build log と artifact metadata に残し、配布 artifact へ暗黙に持ち込まない。artifact metadata には少なくとも次を記録する。

- canonicalized `ROCM_PATH`、ROCm release、compiler path と version
- `CMAKE_HIP_ARCHITECTURES`、`ULLM_HIP_CODEGEN_FEATURES`、code object ABI
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

[GPU 互換性方針](gpu.md) の evidence 値 `vendor-supported`、`project-verified`、`unverified` は、vendor 公式掲載または uLLM 実機検証の根拠を表し、この lifecycle 軸とは役割が異なる。GPU evidence が十分でも OS、runtime library、artifact 条件まで一致しなければ software tuple は `supported` にならない。また、software lifecycle が `experimental` であることから vendor 公式対応の有無を推論してはならない。tuple record は lifecycle と GPU evidence を別 field に保持する。

### 初期候補 tuple

| Lifecycle | Ubuntu | Kernel | ROCm | GPU と artifact 条件 | 備考 |
| --- | --- | --- | --- | --- | --- |
| `experimental` | 24.04.4 LTS | GA 6.8 | build/runtime とも 7.14.0 | GPU、target、features ごとに個別 tuple | 主開発候補。現時点では uLLM 実機検証結果なし |
| `planned` | 26.04 LTS | GA 7.0 | 7.14.0 | GPU、target、features ごとに個別 tuple | 将来検証候補 |

- Ubuntu 24.04.4 LTS、GA kernel 6.8、ROCm 7.14.0 の組み合わせを主系統候補とする。具体的な driver、GPU、target/features、dynamic library path まで確定した tuple だけを evidence の対象にする。
- Ubuntu 26.04 LTS と ROCm 7.14.0 の組み合わせは将来検証する `planned` tuple とする。AMD が ROCm 7.14.0 で Ubuntu 26.04 を掲載していても、uLLM による実機検証なしに Ubuntu 24.04 の結果を移植しない。
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
| canonical GPU | V620 exact `gfx1030`、BDF `0000:03:00.0`、UUID `GPU-76a08c022586fed6`; R9700 exact `gfx1201`、BDF `0000:47:00.0`、UUID `GPU-a8e9ddefa2d60f55` |
| artifact | target別専用binary、Code Object V6、wave32、`xnack`/`sramecc=unsupported`; SHA-256 `gfx1030=40a55e8028355dd1b27b26886ccfef6d0b4085569d2656f90e7ebdc2be1a852c`、`gfx1201=69207b19c1146f73258db848fd5da74a25dd0a8e980b090ee09037da0dd2b1f5` |
| runtime libraries | `/opt/rocm/core-7.14/lib/libamdhip64.so.7.14.60850-0000000`、`/opt/rocm/core-7.14/lib/libhsa-runtime64.so.1.21.0` |
| G0/G1 scope | G0 identity/read-only health/process、G1の1、3、17、255、256、257 byte。各case 2 allocation、2 transfer、1 diagnostic dispatch、byte exact、CPU fallback/model/semantic opなし |

この結果はsemantic数値kernel、full model、性能、generic code object、複数GPU実行、長時間安定性、別GPU/SKU、またはvendor-supported OS/GPU tupleを証明しない。G0/G1 evidenceは検証したcandidate・artifact・canonical 2 GPUへだけ結び付ける。

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
