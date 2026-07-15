# P3 profile actual operator manifest v3

## Scope

- P3 profile diagnosticの単一actual実行を明示的に行うためのoperator manifest v3を作成した。
- この作業ではactual実行、GPU実行、サービス操作を行っていない。
- static `target-command-manifest.json`は含めず、launcherがlive preflight後に実行ごとに生成する契約を固定した。

## Exact command

- cwd: `/home/homelab1/coding-local/ultimateLLM/uLLM-project`
- shell: `false`
- maximum invocations: `1`
- argv SHA-256: `8249eddef3d10029e47f2b47986aa77202c32a080e2f2b05161f3849f37d3271`
- argv:
  - `/usr/bin/python3.12`
  - `/home/homelab1/coding-local/ultimateLLM/uLLM-project/tools/run-aq4-p2-resident-smoke-maintenance.py`
  - `--mode`
  - `execute`
  - `--profile-diagnostic`
  - `--ready-artifact`
  - `/home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-profile-ready-v1/ready-binding.json`
  - `--evidence-output`
  - `/home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-profile-maintenance-evidence-v3`
  - `--confirm-one-case`

## Inputs

- final ready artifacts: commit `eb4840f2f3ddcfb27c0e6e5de259f1b6002f7c07`
- maintenance source: commit `c3a676a962e542b997c14a695328d5cdbfa6c120`
- quiet window v7: commit `189becb0290d1e473b359f362c93ca043b49f7f3`, evidence SHA-256 `5d210245c52248aec489b9ec6820c4d208ad8346cdf09c4a2f3babe4909eb7b6`
- profile-ready `SHA256SUMS`: `98e7a10aded874feddee3f937446fb73f5a987c4a3449cb7a3d2bcb10beef76e`
- launcher/capture: commit `86fb3df33a18ee1b03934a58c7102ddcd6158128`
- runner: commit `ede2b872ab0de5550adbcb1b1dca8b4bbd789efd`
- input set SHA-256: `e699320fdd79524aebdcd7233124da8ab8bb798ed676056892780792880b7092`

## Quiet and fresh boundary

- final streak: sample 24..51、28 samples、200.537321143 seconds
- fixed HEAD/tree: `3023148fd63ef93805837077b574be28478e7f54` / `d057eb82ef0b1b3a21e99c385453edf9c2161e12`
- external process、foreign AMD owner、foreign KFD ownerはfinal streakで各0件である。
- relevant file countは30、byte aggregateは`5f8b91af3bfb90d39ba830b3242aa46e00d30dbc9bfd2d4c10f5aa6dcc349cce`、identity aggregateは`a6051a932743cf29f6d997bb5bc57d9a07d698098ba011601b0142bbb6587759`で固定した。
- formal health identityのstart/end SHA-256は共に`033ce682de4abe6557294b1425a2ac49a8f170bdc3af6bef4409f32d374ba370`である。
- fresh output set SHA-256: `65c2937d825563d6c1d56834f716feb28732ea13d1e1f5b8ad3bff7d63cd69ae`

## Verification

- dry/fake/readback selected tests: `5 passed`
- Git `commit:path` readback: `10/10`
- input SHA-256 readback: `13/13`
- fresh outputs: `5/5 absent`
- manifest self-hash: `c0d169e5ff8831b61df23bba363bb49ab83a148532ffe6863c5482597d54e66a`
- manifest file SHA-256: `073f61e9396245d2c11b49e0846e6868b21185d27c92253447c76fc3e3433c1b`
- `SHA256SUMS` file SHA-256: `a21ac3d1d55272199c4a7cbb7589f0c4dccdbcb2189b044dff4c9bbdf10c409a`
- manifest/SUMSはmode `0444`、ディレクトリはmode `0555`である。
