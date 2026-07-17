# T1 mixed request-state weight-bundle sharing

## 前回の要点

- full mixed request-state smokeはAQ4 payload、AQ4 scale/row-scale、passthrough weight bufferのslot間共有まで通っていた。
- resident layer structはweightとstate/workspaceが混在しており、weight-only resident bundle境界は未分離だった。

## 今回の変更点

- `PackageSelfAttnResidentStepWeights` を追加し、self-attnのimmutable weight/configとrequest slot state/workspaceを分離した。
- `PackageLinearAttnResidentStepWeights` を追加し、linear-attnのimmutable weight/configとrequest slot state/workspaceを分離した。
- batch layer loaderは1slot目でweight bundleを作り、2slot目以降は同じ `Arc<...Weights>` を再利用してstate/workspaceだけを作る。
- paged KV cache、block table、Conv1d history、recurrent state、workspaceはrequest slot別に残した。

## 検証

- `cargo fmt --all --check`
- `cargo check -p ullm-engine`
- `cargo test -p ullm-engine request_slot_index -- --test-threads=1`
- `cargo build -p ullm-engine`
- `jq empty benchmarks/results/2026-07-08/sq-r9700-state-freeze-v0.1.json`
- `git diff --check -- ':!README.md'`
- R9700 `package-token-ids-mixed-request-state-smoke ... manifest-all len:2x2 ...`

R9700 smoke result:

- `verified=true`
- `final_top1_tokens=44370,5446`
- `self_attn_weight_bundle_shared=true`
- `linear_attn_weight_bundle_shared=true`
- `layer_load_ms=10256.735321`
- `total_wall_ms=10907.692026`

## 次の行動

1. request-state interleaved stepをreal request-batch executorへ置き換える。
2. full packageで `batching.mode=real`、`prefill_real_batch=true`、`decode_real_batch=true` のAQ4 baseline rowを保存する。
3. T2 SQ候補を同じfull mixed pathへ接続する。
