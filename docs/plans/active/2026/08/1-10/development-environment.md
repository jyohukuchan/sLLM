# 初期開発環境構築計画

## 目的

repository skeletonと初期CIへ進む前に、基準toolchainをこの開発hostへ導入し、Rust、C++17、ROCm/HIPをCPU代替なしで検証できる再現可能な入口を用意する。

## 作業単位と受入条件

- Rust開発pin 1.97.1とMSRV 1.85.0を利用でき、repository内では`rust-toolchain.toml`により1.97.1が自動選択される。
- G++でC++17 sourceをwarning error付きでcompile・実行できる。
- CMake 3.21以上とNinjaを利用できる。
- ROCm compiler、headers、runtime、device librariesを`/opt/rocm/core-7.14`の単一rootから解決し、AMD clang LLVM major 23を使用する。
- exact `gfx1030`と`gfx1201`を含むHIP binaryをcompileし、各visible GPUで実kernelを実行する。
- GPU不在、compile失敗、runtime失敗、target不一致をCPU fallbackまたはskipで成功扱いにしない。
- local smokeの結果を正式なG0/G1、full model、数値正しさ、性能、対応target全体のevidenceへ拡大解釈しない。
- tracked script、toolchain pin、文書を検証し、同じcandidateを正本へ反映してpushする。

## 導入方針

- systemに既にあるG++、CMake、Ninja、Python、uv等はversionと動作を確認して利用する。
- Rustはrustupのuser toolchainとして導入する。
- ROCmはAMD公式APT source `https://repo.amd.com/rocm/packages-multi-arch/ubuntu2404` の `stable main` からsystem packageとして導入し、rootは`/opt/rocm/core-7.14`とする。
- `amdrocm-core-sdk7.14-gfx1030`と`amdrocm-core-sdk7.14-gfx1201`をpackage version `7.14.0-3`で導入する。
- legacy ROCm user-space packages、旧installation root、旧ROCm APT sourceを除去し、amdgpu driver packagesは変更せず保持する。
- runfileとmulti-architecture tarballは移行前の検証履歴として保持するが、現行の導入方式にはしない。
- runfileはcustom targetでも内部copyにsudoを要求したため採用しない。passwordをfile、stdin、argv、環境変数へ渡さない。
- tarball、展開済みSDK、build output、raw logはGit管理しない。

## 実施状況

- [x] Ubuntu、kernel、amdgpu、GPU、既存toolchainとsystem ROCmをread-onlyで監査。
- [x] Rust 1.97.1、rustfmt、clippyとMSRV Rust 1.85.0をrustupへ導入。
- [x] clang-format 22.1.8とShellCheck 0.11.0をuser領域へ導入。
- [x] 旧ROCm user-space packages、installation root、APT sourceを除去し、amdgpu driver packagesを保持。
- [x] AMD公式APT sourceからROCm SDK packageを導入し、`/opt/rocm/core-7.14`をcurrent rootとして確認。
- [x] `amdrocm-core-sdk7.14-gfx1030`と`amdrocm-core-sdk7.14-gfx1201`のpackage version `7.14.0-3`を確認。
- [x] C++17 host compile/runを確認。
- [x] ROCm 7.14.0、LLVM 23、HIP headers、runtime libraryの同一root解決を確認。
- [x] exact `gfx1030,gfx1201` fat binaryをcompileし、visible GPU 3台で最小HIP kernelを実行。
- [x] tracked activation/check scriptとprobe sourceを追加し、static check、host check、GPU smokeを再実行。
- [x] compatibility文書、main plan、historyを最終evidenceへ同期し、公開candidateを作成。

## 現時点の実機結果

- OS: Ubuntu 24.04.4 LTS、kernel `6.17.0-35-generic`。
- amdgpu: `6.16.13`。
- ROCm: 7.14.0、AMD clang 23.0.0git、HIP runtime `71460850`、root `/opt/rocm/core-7.14`。
- packages: `amdrocm-core-sdk7.14-gfx1030=7.14.0-3`、`amdrocm-core-sdk7.14-gfx1201=7.14.0-3`。sourceは`https://repo.amd.com/rocm/packages-multi-arch/ubuntu2404`の`stable main`。
- GPU: Radeon Pro V620 `gfx1030` 2台、Radeon AI PRO R9700 `gfx1201` 1台。
- smoke: 各deviceでallocation、copy、kernel dispatch、synchronize、copy-back、freeを実行し、入力41から出力42を確認。
- loader: `libamdhip64.so.7`と`libhsa-runtime64.so.1`はsystem packageの`/opt/rocm/core-7.14`から解決。

## 未完了・非証明範囲

- このhostは初期候補のUbuntu 24.04.4 GA kernel 6.8ではなくHWE 6.17であり、mixed V620/R9700構成全体のvendor support判定は未完了。
- codegen feature、Code Object version、generic target、artifact metadata、capability profile、resource gateは未固定。
- 正式なG0/G1 report schema、runner isolation、CI artifact identityはrepository skeletonで実装する。
- 数値kernel、model slice、Qwen3.5、API、長時間安定性、性能は未検証。

[対応する履歴](../../../../../history/2026/08/1-10/development-environment.md)
