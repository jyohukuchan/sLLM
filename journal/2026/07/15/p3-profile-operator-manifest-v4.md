# P3 profile actual operator manifest v4

## Scope

- profile diagnostic v4の単一actual実行に使用するoperator manifestだけをfresh作成した。
- authorityはofficial artifact commit `4cadba39e1310f28552abcbe55c88425149d6614`、maintenance source commit `92e7c6ee76ebb0ef44d41f928eb8a93131146817`である。
- quiet window v8はcommit `098e561192fb07afb3dfca0f0868d607150ab88e`、evidence SHA-256 `c48ed2885b7b53e8e9ee62fcd2f67274ee54329b209e56e577e603afbf49d85b`で固定した。
- この作業ではactual実行、GPU実行、モデルロード、sudo、サービス操作を行っていない。

## Exact command

- cwd: `/home/homelab1/coding-local/ultimateLLM/uLLM-project`
- shell: `false`
- maximum invocations: `1`
- argument count: `10`
- argv SHA-256: `d066444a6cc7fb526c67232e0e74e33732d85938ac120dcfd6bc55e01622be9c`
- argv:
  - `/usr/bin/python3.12`
  - `/home/homelab1/coding-local/ultimateLLM/uLLM-project/tools/run-aq4-p2-resident-smoke-maintenance.py`
  - `--mode`
  - `execute`
  - `--profile-diagnostic`
  - `--ready-artifact`
  - `/home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-profile-ready-v1/ready-binding.json`
  - `--evidence-output`
  - `/home/homelab1/coding-local/ultimateLLM/uLLM-project/benchmarks/results/2026-07-15/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-profile-maintenance-evidence-v4`
  - `--confirm-one-case`

## Fresh and quiet boundary

- 4つのfresh rootに加え、implicit `capture-artifact.json`、`rocprof.stdout`、`rocprof.stderr`を個別に列挙し、7/7の不在を確認した。
- quiet final streakはsample 0..28の29 samples、137.175538188秒である。
- HEAD/treeは`4cadba39e1310f28552abcbe55c88425149d6614` / `93588f9642eb6bafaafed497c641c4844f5557f0`で固定した。
- relevantは50 files、byte aggregate `98dc3cc51efdfdd8e719a3e67c06a34c2c18fb88864c410a603e5211e930d8f9`、identity aggregate `013a6612c2b4e30adeba59ce2fe6e2c16570a61bf867c3feb556695d19dd7563`である。
- formal-health identityのstart/endは共に`4c762917af1eb0dc016a87d1494fd89f7518df668e6d0097f035b8083a95c1f2`である。
- external process、foreign AMD owner、foreign KFD ownerは各0件で、AMD/KFD ownerは共にworker PID `3213208`だけである。
- static target command manifestは存在せず、launcherがlive preflight後に実行ごとに生成する。

## Verification

- dry/fake/readback selected tests: `5 passed`
- Git `commit:path` readback: `10/10`
- input SHA-256 readback: `13/13`
- fresh outputs: `7/7 absent`
- v3 actual failureはcommit `ec4b0a36e9f10db524cb24ef2b2d5e3bf638249d`から不変である。
- secret scanとmanifest semantic self-hash検証はpassした。
- input set SHA-256: `501b031b51ecd4bccbc9547827dbf969ecaf7f2acae3bde0d4876ee242e8af93`
- fresh output set SHA-256: `c1f436c4e7ba443c8e77185e942efe43684e9832bb105028175163eba9c2ebad`
- manifest self SHA-256: `a3b4c499e0151bb7dd1bbb166a8afc535d27e77cdcdf5b42a9ab83dfcdbe994a`
- manifest file SHA-256: `97f4ea10353575dedbe23c642fc0f6bd6391899ef969305eb5442cc30b38baad`
- `SHA256SUMS` file SHA-256: `ccd64a86d36a12bf8a6604b7b291d446556c78a9e2d2160b22f9b84ff19fd509`
- manifest/SUMSはmode `0444`、ディレクトリはmode `0555`である。
