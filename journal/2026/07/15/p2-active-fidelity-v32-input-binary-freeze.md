# P2 active fidelity v32 input and capture binary freeze

## 前回の要点

source側は v32 の24行CPU captureを完了し、source artifact validatorを通過した。source producerの成功基準は `05a8ab661b8e56559353f5a530ec8abac08b9a68` の証跡へ固定した。active側のservice停止、GPU実行、出力生成はまだ行っていない。

## 今回の変更点

active gateの入力をv32へ固定し、旧16-thread、旧plan、cases別パス、source/build placeholderを解消した。

- source artifact: `attempts/source-attempt-v32-20260714T180609Z/source-full`
- source SHA256SUMS SHA: `6d27caef27dabf02dcc56b0b298290f9811355ba36c34e6c9d23939baf50edde`
- source manifest SHA: `78a6de7d2cae4c2ff31952cfe345fefbce55dfd67db7a4904ba10f4e5f7438bc`
- plan SHA: `1b4f8c244e922ab73c0bb026216d8333a9cfe57c23e6695c4141554d117693c0`

capture binaryは、汚れていない専用worktree `/tmp/ullm-aq4-fidelity-build-v32` のcommit `05a8ab661b8e56559353f5a530ec8abac08b9a68` から、次の低並列コマンドでbuildした。

```text
CARGO_BUILD_JOBS=1 cargo build -p ullm-engine --release --bin ullm-aq4-fidelity-capture
```

build receiptはtree SHA `12e6d777f37d648ede369263296cd5606676a441`、clean=true、Cargo.lock SHA `10df8371ae3a33ed792dc4e8c15dd6196a8a7e176e377ef275e75b3219aa157b` を記録する。active inputへinstallしたbinaryはSHA `82c878a4974cdbc442458c6b3366b0eae20d355896d8b18d5d76fe311c0b083e`、3,200,896 bytes、mode 0755、nlink 1。receipt SHAは `3d09df92aa2bef098c8c64ef7bcd63ed0b23dd2160a44dfa3799421477440ede`、build log SHAは `b1572ccc0333dcef12b5c089a6955a804af08a351576b1fa6e9f42c45926614a` である。gateはbinary内のbuild commit文字列ではなく、receipt、tree/commit、Cargo.lock、binary SHA、SHA256SUMS、nlink1を検証する。

GPU/service操作なしで、次を実施した。

- `bash -n` gate script
- `MOCK_PREFLIGHT=1 PREFLIGHT_ONLY=1` gate（return 0、service_stop=0、gpu_run=0）
- `python3 -m unittest -v tests/test_qwen35_aq4_active_fidelity_gate_template.py`（4 tests passed）
- binary/source `sha256sum -c SHA256SUMS`（全項目OK）
- source artifact validator（status valid、row_count 24、nonfinite_rows 0）
- 対象差分の `git diff --check`

## 次の行動

親が限定commitを再監査した後、必要ならmock/locked read-only preflightを再実行する。本番service停止またはGPU captureは、このfreezeと復旧条件を確認した明示判断の後だけ行う。binary、source artifact、output/logの既存証跡は上書きせず、preflight失敗時はその場で停止する。
