# AQ4 P2 fidelity split evidence (measurement-free)

- 実行日時: 2026-07-15 JST。モデル起動、GPU操作、service操作、calibration metrics/freezeは行っていない。
- fixture binding root: `benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/fidelity-holdout-fixture-binding-v0.1/`
- split root: `benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/fidelity-holdout-split-v0.1/`
- 入力trust SHA: expanded v2 `427bd765cc8ce56d95ee3414a2a5cd9a39309929f1c2a5198a4786aeb189ff4c`（source case manifest SHA `1fa264c6a7a485e36b1119ca13732ad88e052a8bd502c2addacdff14ff41cbea`）、served-model `feb3190d0ff59778e4da140b8db2bd1ce2ba440e3a69e844b997011d4d08cb44`、fixture-index `26d22e4f209c482e0479dba67525149d6a3e38a3897d721c47b3d4b0ed04aec1`。
- split files: `calibration-cases.jsonl` SHA `20c09f22bb1ca4dfac907de09febddb01ed0228c3f4a17c01efd646491e0983f`、`holdout-cases.jsonl` SHA `7e5a87ff103f54b754659828493b066da495708aaa2596a7f9ca9917bfde27a9`、`policy.json` SHA `302c3219af286a970ddf39ed090021ef102b51b2d188c0ff337f6b9dd04d1a03`、`split-manifest.json` SHA `966878f3d9eb13f5b485825208f8072521724f308f5ee3d8a003b0b051198887`、split `SHA256SUMS` SHA `bba101a39167608f6fa543066605a5d02f875844bba55d9ee6f8165d5a14d8fc`。
- fixture bindingのSHA256SUMS SHA `fd34ec72df10a0ebd852b421ba0b19a54839955a6f9fef6303754f856298a1b2`（trust SHA 3件、index 1件、fixture 48件を含む52行）。fixtureは48件、prompt/context/fixture hashは各subset内で24件ずつ一意である。
- 校正/holdoutは各24件、8 strata（prompt 1011/1024/1339/2048 × `all_m1`/`cold_batched`）を各3件ずつ含む。各行は `cached_prefix_tokens=0`、`context_tokens=prompt_tokens`、`generated_tokens=0`、`step=0`、`row_count=1`。attempt2のID/context hashは0件である。
- mode分布は各subset `all_m1=12,cold_batched=12`、prompt分布は各subset `1011/1024/1339/2048=6`。M分布はcalibration `1:3,8:5,16:3,32:5,64:5,128:3`、holdout `1:5,8:3,16:5,32:3,64:3,128:5`。
- generator再実行は `/tmp/fidelity-split-second` へ行い、5ファイル（calibration/holdout/policy/manifest/SHA256SUMS）が `cmp` byte-identical、validatorもrc0だった。fixture pathは絶対pathへ正規化し、validatorがfixture実体・prompt/context hashまで再検証する。
- split generator実測資源（`/usr/bin/time -v`）: 初回 wall `0.23 s`、user `0.17 s`、sys `0.04 s`、最大RSS `54,620 kB`。validator wall `0.08 s`、user `0.07 s`、sys `0.01 s`、最大RSS `17,792 kB`。これはCPUのsplit/検証だけの値で、モデル時間やGPU時間ではない。
- split rootにはmetrics/receiptが存在せず、holdout評価は未開始である。
