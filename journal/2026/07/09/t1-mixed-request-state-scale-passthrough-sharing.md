# T1 mixed request-state scale/passthrough sharing

## 前回の要点

- `package-token-ids-mixed-request-state-smoke` は `manifest-all` full mixed orderで通っていた。
- AQ4 index/scale/codebook payloadはslot間共有済みだった。
- AQ4 `scale_values_buffer`、row-scale、passthrough weight bufferはまだrequest slotごとに確保していた。

## 今回の変更点

- `PackageResidentSharedBufferRegistry` を追加し、batch layer内のrequest slot間でf32 runtime bufferを共有できるようにした。
- AQ4 `scale_values_buffer` / row-scale bufferも `load_with_shared_buffers` 経由で共有した。
- self-attnはinput/q/k/post RMSNorm weightを共有した。
- linear-attnはinput/post RMSNorm、linear-attn norm、Conv1d weight、A_log、dt_biasを共有した。
- workspace、Conv1d history、recurrent state、paged KV cache、block tableはrequest slot別に残した。

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
- `slot_aq4_payload_registry_shared=true`
- `slot_aq4_scale_values_shared=true`
- `slot_passthrough_weight_buffers_shared=true`
- `layer_load_ms=16240.561131`
- `total_wall_ms=16859.363507`

## 次の行動

1. weight-only resident bundleをlayerごとに1つへ寄せる。
2. request-batch stepをreal batch executorへ置き換えてfull package throughput rowを作る。
3. SQ候補をfull mixed pathへ接続する準備を続ける。
