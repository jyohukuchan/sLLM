# 初期開発環境構築履歴

## 2026-08-02

- hostを監査し、Ubuntu 24.04.4、kernel `6.17.0-35-generic`、amdgpu `6.16.13`、system ROCm 7.2.1、Rust 1.96.0を確認した。
- system G++ 13.3.0、CMake 3.28.3、Ninja 1.11.1、Python 3.12.3、uv、jq、git、ghはそのまま利用可能と判定した。
- rustupへ開発pin 1.97.1とMSRV 1.85.0を導入し、1.97.1へrustfmtとclippyを追加した。
- repository rootへ`rust-toolchain.toml`を追加し、1.97.1の自動選択を確認した。
- clang-format 22.1.8をuv toolとして、ShellCheck 0.11.0を公式release archiveからuser領域へ導入した。
- ROCm 7.14.0 runfileを検証したが、異種GPUの自動検出とcustom targetへのcopyで内部sudoが必要になったため採用しなかった。passwordは渡さず、system ROCmとdriverは変更していない。
- AMD公式`therock-dist-linux-multiarch-7.14.0.tar.gz`を取得し、8,522,482,978 bytes、SHA-256 `baadd54cff7a064b3b0ae51c19606ee4bced0f4215a21c89f616cf9c01ea4b47`を記録した。
- tarballを`$HOME/.local/amd-rocm/7.14.0`へ展開し、`$HOME/.local/amd-rocm/current` symlinkを設定した。
- ROCm 7.14.0、AMD clang LLVM 23、HIP runtime `71460850`と、runtime libraryが同一rootから解決されることを確認した。
- exact `gfx1030,gfx1201` fat binaryをC++17/HIPでcompileし、Radeon Pro V620 2台とRadeon AI PRO R9700 1台の全visible deviceで入力41から出力42になる最小kernelを実行した。CPU fallbackは使用していない。
- `scripts/dev/activate-rocm.sh`、`check-environment.sh`、`hip-smoke.cpp`を追加し、ROCm rootのfail-closed選択、toolchain検査、全visible GPU実行を再現可能にした。
- Bash syntax、ShellCheck、clang-format、C++17 host probe、ROCm root選択の冪等性、HIP/HSA runtimeの同一root解決、3台のGPU smokeを検証した。
- system ROCm 7.2.1の明示選択と`gfx1201` target欠落が成功扱いにならないnegative testを実行した。
- このsmokeを正式なG0/G1、full model、数値正しさ、性能、またはtarget全体の互換性evidenceとして扱わないとした。

[対応する計画](../../../../plans/active/2026/08/1-10/development-environment.md)
