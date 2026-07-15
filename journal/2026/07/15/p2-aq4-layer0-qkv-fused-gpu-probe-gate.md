# P2 AQ4 layer-0 QKV/Z/gate/beta fused GPU probe gate

## 前回の要点

fused probe report schema v2とCPU診断は既存実装で完了していた。HIP GPU、service停止、holdout、数値Go/No-Go、promotionは未実施である。

## 今回の変更点

- clean detached worktree `6082df4966190ae4977b699460a5ecb93fee8e34`から、`CARGO_BUILD_JOBS=1 cargo build --release -p ullm-engine --bin ullm-aq4-layer0-qkv-z-gate-beta-runtime-probe`を実行した。
- 新artifact root `benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/aq4-layer0-qkv-fused-gpu-probe-v0.1/input/probe-binary-v0.1/`にnlink=1、mode=0555のimmutable probe copy、build receipt、`SHA256SUMS`をno-overwriteで保存した。
  - binary SHA-256: `42752e7a29614f59f72f90bed6797c3e925b032bffb1a4196c462c8476386840`
  - receipt SHA-256: `90e9ef6d383f7ef25e9526659f035e40291ba1a5efa7f8ba36340c8b245d9504`
- fused gateは既存standalone gateを最小コピーして、read-only preflight/mockを既定にした。execute時はread-only prestate→service stop→stable2 old worker/GPU/KFD消滅→sudo `install -d`→exact lock inode acquire→fused probe→service停止中のlock unlink+rmdir→service start→new epoch、health、lock owner検証の順序を固定した。
- gateのHIP契約は`HIP_VISIBLE_DEVICES=1`、`ULLM_HIP_VISIBLE_DEVICES=1`、両AQ4 matvec guard=1、dedicated fused RPB=4、standalone RPB unset/default32記録、device 1、gfx1201、5 sidecars+reportのfinite/layout/identityをfail closedで検証する。promotionは常にfalseである。

## 検証

- 実行: `bash -n`、release build、binary/receipt SHA256SUMS検証、`PREFLIGHT_ONLY=1`、`MOCK_PREFLIGHT=1`、新規gate unittest。
- 未実施: GPU probe、service stop/start、ROCm SMI/KFD実測、health実測、holdout、数値閾値判定、promotion。

## 次の行動

この限定commitを親worktreeへcherry-pickする。君の明示的なGPU/service実行許可までは、preflight/mockと静的契約検証だけを行う。
