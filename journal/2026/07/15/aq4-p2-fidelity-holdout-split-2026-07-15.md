# AQ4 P2 fidelity split evidence (measurement-free)

- 実行日時: 2026-07-15 JST。モデル起動、GPU操作、service操作、calibration metrics/freezeは行っていない。
- fixture binding root: `benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/fidelity-holdout-fixture-binding-v0.1/`
- split root: `benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/fidelity-holdout-split-v0.1/`
- 入力trust SHA: expanded v2 `427bd765cc8ce56d95ee3414a2a5cd9a39309929f1c2a5198a4786aeb189ff4c`（source case manifest SHA `1fa264c6a7a485e36b1119ca13732ad88e052a8bd502c2addacdff14ff41cbea`）、served-model `feb3190d0ff59778e4da140b8db2bd1ce2ba440e3a69e844b997011d4d08cb44`、fixture-index `26d22e4f209c482e0479dba67525149d6a3e38a3897d721c47b3d4b0ed04aec1`。
- split files: `calibration-cases.jsonl` SHA `f9ad1e61c41adcd78c0be77bca0c85d34e3a668efaf728b7f996298f8e6469dc`、`holdout-cases.jsonl` SHA `7eace033220e523b41a33d4ec7b10a24294843e03ab63432093b2e3637fb5ad2`、`policy.json` SHA `302c3219af286a970ddf39ed090021ef102b51b2d188c0ff337f6b9dd04d1a03`、`split-manifest.json` SHA `f243c02654b0450187e17fb48b363428777944c4cfbeabd1aa1ad970bfa6673b`、split `SHA256SUMS` SHA `bdd2fcd50986145820f4d8b2de8a5a7df560410b54620c43f802208cfb15c7d2`。
- fixture bindingのSHA256SUMS SHA `fd34ec72df10a0ebd852b421ba0b19a54839955a6f9fef6303754f856298a1b2`（trust SHA 3件、index 1件、fixture 48件を含む52行）。fixtureは48件、prompt/context/fixture hashは各subset内で24件ずつ一意である。
- 校正/holdoutは各24件、8 strata（prompt 1011/1024/1339/2048 × `all_m1`/`cold_batched`）を各3件ずつ含む。各行は `cached_prefix_tokens=0`、`context_tokens=prompt_tokens`、`generated_tokens=0`、`step=0`、`row_count=1`。attempt2のID/context hashは0件である。
- mode分布は各subset `all_m1=12,cold_batched=12`、prompt分布は各subset `1011/1024/1339/2048=6`。M分布はcalibration `1:3,8:5,16:3,32:5,64:5,128:3`、holdout `1:5,8:3,16:5,32:3,64:3,128:5`。
- generator再実行は `/tmp/fidelity-split-second` へ行い、5ファイル（calibration/holdout/policy/manifest/SHA256SUMS）が `cmp` byte-identical、validatorもrc0だった。
- split generator実測資源（`/usr/bin/time -v`）: 初回 wall `0.22 s`、user `0.15 s`、sys `0.04 s`、最大RSS `54,636 kB`。再実行 wall `0.23 s`、user `0.16 s`、sys `0.04 s`、最大RSS `55,644 kB`。これはCPUのsplit/検証だけの値で、モデル時間やGPU時間ではない。
- split rootにはmetrics/receiptが存在せず、holdout評価は未開始である。
