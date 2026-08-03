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

[対応する計画](../../../../plans/active/2026/08/1-10/phase2-h3-g0-model-free-gpu.md)
