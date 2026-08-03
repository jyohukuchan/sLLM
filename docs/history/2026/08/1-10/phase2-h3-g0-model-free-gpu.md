# Phase 2 H3・G0・model-free GPU path履歴

## 2026-08-03

- Phase 1完了後の次作業を、ROCm 7.14.0固定toolchain、exact `gfx1030`/`gfx1201` H3、trusted local GPU evidence、G0、model-free最小GPU実行までに限定した。
- H3の20回以上・7日以上の観測はrequired昇格だけの条件であり、G0とmodel-free pathの開発を停止しないと決定した。
- 現行GPU hard gateが未構築のG0/G1/G2/G4/P0をH3自身へ要求するbootstrap循環を記録し、実装前にscope別gateへ修正する作業単位を計画の先頭に置いた。
- model-free最小経路を`Cargo -> ullm-hip -> versioned C ABI -> native HIP -> GPU`とし、allocation、copy、diagnostic kernel、completion、copy-back、解放をcanonical `gfx1030`/`gfx1201`で検証する到達点を定めた。
- 数値op、model load・推論、性能、generic target、互換性昇格を計画範囲外とした。
- この時点では計画文書だけを作成しており、H3、G0、GPU runtimeの実装evidenceはまだない。
- 作業単位0としてCI hard gateを変更scope別へ分割し、H3、G0 runner、model-free runtimeに適用する同一candidate evidenceを明確化した。
- H3 required昇格観測とG0/model-free実装を並行するとCI正本へ同期した。
- `gpu.md`の「実機検証結果なし」を、exact `gfx1030`/`gfx1201`の限定smokeだけが存在しformal G0/G1以降は未検証という表現へ修正し、AMD/software文書と整合させた。
- 公式`docker.io/rocm/dev-ubuntu-24.04:7.14.0-full`をsingle `linux/amd64` manifest digest `sha256:439edaa8f0c4be4a3728e528f87b8a2ea1f051f34cf10b27caa4bd94f562eda7`とconfig digestで固定し、ROCm 7.14.0、LLVM 23、`/opt/rocm`同一rootを静的contractへ記録した。
- H3 matrixをexact `gfx1030`/`gfx1201`の2 row、Code Object V6、wave32、`xnack`/`sramecc=unsupported`、non-required、compile-onlyとして固定した。
- HIP artifact metadataをhost側のx86-64 offload bundleと抽出後のAMDGPU device code objectへ分離し、bundle identity、target別ELF ABI/e_flags、candidate SHA/tree、manifest hash、artifact size/hash、row-private build path、非実行scopeをfail-closed検証するschemaとvalidatorを追加した。
- tag-only/`latest`、digest/platform/root/version/LLVM不一致、missing/duplicate/unknown/generic/multiple/wrong target、required化、codegen不一致、target差し替え、stale identity/hash、source/shared build出力、誇張した実行scopeを拒否するnegative testを追加し、255/256/257 byte境界も確認した。この作業単位は静的contractだけであり、H3 compile、GPU実行、数値・model・性能evidenceはまだ生成していない。

[対応する計画](../../../../plans/active/2026/08/1-10/phase2-h3-g0-model-free-gpu.md)
