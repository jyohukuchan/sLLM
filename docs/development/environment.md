# 開発環境の構成と検証

## 最小構成

開発用toolchainはRust 1.97.1、MSRV確認用Rust 1.85.0、C++17 compiler、CMake 3.21以上、Ninja、clang-format、ShellCheck、ROCm 7.14.0同梱のLLVM 23を必要とする。ROCmのcompiler、headers、CMake package、device libraries、HIP runtimeは一つのROCm rootから解決し、別releaseへ暗黙にfallbackさせない。

ローカル開発用のROCm環境はrepository rootで次のように有効化する。

```bash
source scripts/dev/activate-rocm.sh
```

`activate-rocm.sh`は次の優先順位で候補を一つだけ選ぶ。

1. 明示的な`ULLM_ROCM_PATH`
2. 既存の`ROCM_PATH`
3. `$HOME/.local/amd-rocm/current`

選んだ候補が存在しない、canonicalizeできない、ROCm 7.14.0でない、または`amdclang++`がLLVM major 23でない場合は失敗する。一つの候補を選んだ後で別releaseへfallbackしない。成功時はcanonical rootを`ROCM_PATH`と`HIP_PATH`の両方へ設定し、そのrootの`bin`、`llvm/bin`、`lib`を`PATH`と`LD_LIBRARY_PATH`へ重複なしで優先配置する。compilerは安定したentry pointである`$ROCM_PATH/bin/amdclang++`を使用し、symlink解決後の実体も同じroot内にあることを検査する。

一時的に別の展開先を使う場合は、source前に明示する。

```bash
export ULLM_ROCM_PATH=/absolute/path/to/amd-rocm/7.14.0
source scripts/dev/activate-rocm.sh
```

## 環境検査

GPUを使わないhost契約は次で検査する。

```bash
scripts/dev/check-environment.sh host
```

このmodeはRust 1.97.1、MSRV toolchain 1.85.0、CMake 3.21以上、Ninja、C++17のcompileと実行、clang-format、ShellCheck、ROCm 7.14.0、LLVM 23、およびROCm componentが同じcanonical rootにあることをfail-closedで確認する。

実GPU smokeは次で実行する。

```bash
scripts/dev/check-environment.sh gpu
```

`gpu` modeはhost検査を先に行い、`hip-smoke.cpp`を`amdclang++`のHIP modeで明示targetを含むfat binaryへcompileする。既定targetは`gfx1030,gfx1201`である。別のexact target集合はcomma-separated valueで指定できる。

```bash
ULLM_HIP_ARCHITECTURES=gfx1030,gfx1200,gfx1201 \
  scripts/dev/check-environment.sh gpu
```

probeはHIP runtimeが列挙した全visible deviceについて、device allocation、host-to-device copy、kernel launch、synchronize、device-to-host copyを実行し、結果が42であることを検査する。runtime versionと、各deviceのname、exact `gcnArchName`、結果を出力する。GPU不在、visible deviceに対応する明示targetの欠落、compile error、runtime error、結果不一致はすべて失敗であり、skipやCPU fallbackにはしない。生成binaryは安全に作成した一時directoryだけに置き、終了時に削除する。

この小さなprobeの成功は、ローカルtoolchainと実機経路のbring-up確認に限る。この限定smoke scopeの`project-verified` evidenceには使用できるが、CI方針上の正式なG0 preflight、G1 kernel/ABI promotion、target全体または別SKUの互換性昇格根拠には単独で使用しない。

## 2026-08-02のhost導入実績

このhostでは次の開発環境を導入し、上記smokeを実行した。

- Ubuntu 24.04.4、kernel `6.17.0-35-generic`
- Rust 1.97.1、MSRV確認用Rust 1.85.0
- G++ 13.3、CMake 3.28.3、Ninja 1.11.1
- clang-format 22.1.8、ShellCheck 0.11.0
- ROCm 7.14.0 tarballを`$HOME/.local/amd-rocm/7.14.0`へ展開し、`$HOME/.local/amd-rocm/current` symlinkから選択
- tarball SHA-256: `baadd54cff7a064b3b0ae51c19606ee4bced0f4215a21c89f616cf9c01ea4b47`
- ROCm同梱LLVM 23、HIP runtime version `71460850`
- `gfx1030` AMD Radeon Pro V620 2台、`gfx1201` AMD Radeon AI PRO R9700 1台でallocation/copy/kernel/sync/copy-back smoke成功

ROCm runfile installerはcustom targetを指定しても内部で`sudo rsync`を実行するため、この非特権なlocal配置には使用しない。tarballを固定hashで検証して展開し、versioned directoryと`current` symlinkを分ける。

この実績だけからOS、kernel、driver、ROCm、GPU SKUの直積を対応済みとは扱わない。kernel/driver/vendor supportを含む完全なcompatibility tuple、正式なG0/G1 evidence、および互換性lifecycleの昇格は別途評価・記録する。
