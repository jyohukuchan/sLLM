# T1 mixed request-state scale/passthrough sharing v1

## 前回の要点

- `package-token-ids-mixed-request-state-smoke` は `manifest-all` で32層full mixed orderまで通っていた。
- 直前の改善で、同一layer内のrequest slot間ではAQ4 index/scale/codebook payload bufferを共有できるようになった。
- ただしAQ4 `scale_values_buffer`、row-scale buffer、RMSNorm/Conv1d/A_log/dt_biasなどのpassthrough weight bufferはまだslotごとに確保していた。

## 今回の変更点

- `PackageResidentSharedBufferRegistry` を追加し、同一batch layer内のrequest slot間でf32 runtime bufferを共有するようにした。
- `PackageAq4ResidentMatvec::load_with_shared_buffers` をbatch loaderから使い、AQ4 `scale_values_buffer` とrow-scale bufferもslot間共有へ寄せた。
- `PackageSelfAttnResidentStepLayer` はinput/q/k/post RMSNorm weight bufferを共有する。
- `PackageLinearAttnResidentStepLayer` はinput/post RMSNorm、linear-attn norm、Conv1d weight、A_log、dt_bias bufferを共有する。
- requestごとのworkspace、Conv1d history、recurrent state、paged KV cache、block tableは引き続きslot別に残した。
- smoke出力に `slot_aq4_scale_values_shared=true` と `slot_passthrough_weight_buffers_shared=true` を追加した。

## R9700 smoke

Command:

```text
target/debug/ullm-engine package-token-ids-mixed-request-state-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d 2 1048576 manifest-all len:2x2 1 1 1024 32 10000000 0
```

Result:

| field | payload shared | scale/passthrough shared |
| --- | ---: | ---: |
| layers | `0..31` | `0..31` |
| requests | 2 | 2 |
| final top1 tokens | `44370,5446` | `44370,5446` |
| slot AQ4 payload registry shared | `true` | `true` |
| slot AQ4 scale values shared | n/a | `true` |
| slot passthrough weight buffers shared | n/a | `true` |
| layer load ms | 17828.586114 | 16240.561131 |
| total wall ms | 18452.035174 | 16859.363507 |
| prefill tok/s | 38.064111 | 38.223872 |
| decode tok/s | 81.410307 | 81.553350 |
| verified | `true` | `true` |

## 判断

- 同一layer内のrequest slot間で、AQ4 payload、AQ4 scale/row-scale、主要passthrough weight bufferを共有する境界は通った。
- final top1 tokenは前回のmanifest smokeおよびpayload共有後と一致した。
- `layer_load_ms` はpayload共有後からさらに約 `1588.024983 ms` 短縮した。
- これはまだ完全なweight-only resident bundleではない。workspace、Conv1d history、recurrent state、paged KV cache、block tableはrequest slot別である。
- `prefill_real_batch=false` / `decode_real_batch=false` のままなので、SQ/vLLM throughput比較には使わない。

## 次の行動

1. weight-only resident bundleをlayerごとに1つへ寄せ、slot別state/cache/workspaceとの境界を明示する。
2. その後、request-batch stepをreal batch executorへ置き換え、full package throughput rowを作る。
3. T2側はこの境界を前提に、SQ候補をfull mixed pathへ接続する準備を続ける。
