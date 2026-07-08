# T1 mixed request-state AQ4 payload sharing v1

## 前回の要点

- `package-token-ids-mixed-request-state-smoke` は `manifest-all` で32層full mixed orderまで通った。
- ただしrequest slotごとにresident layerをロードしており、AQ4 payloadもslotごとに再ロードしていた。
- full package real throughputへ進むには、少なくとも同一layer内のrequest slot間でweight payloadを共有する必要があった。

## 今回の変更点

- `PackageAq4ResidentMatvec::load` が、同じ `WeightRegistry` 内に既に同名tensorがある場合、そのloaded tensor bundleを再利用するようにした。
- `PackageLinearAttnResidentStepBatchLayer` と `PackageSelfAttnResidentStepBatchLayer` は、request slot生成時に同じ `WeightRegistry` を渡すようにした。
- これにより、同一layer内のrequest slot間でAQ4 index/scale/codebook runtime bufferが共有される。
- smoke出力に `slot_aq4_payload_registry_shared=true` を追加した。

## R9700 smoke

Command:

```text
target/debug/ullm-engine package-token-ids-mixed-request-state-smoke /tmp/ullm-quant-direct-package-fullpkg-qwen35-9b-p4p6-lmhead-aq4-g8-weighted-lmhead-calib32-proto.ullm.d 2 1048576 manifest-all len:2x2 1 1 1024 32 10000000 0
```

Result:

| field | before | after |
| --- | ---: | ---: |
| layers | `0..31` | `0..31` |
| requests | 2 | 2 |
| final top1 tokens | `44370,5446` | `44370,5446` |
| slot AQ4 payload registry shared | n/a | `true` |
| layer load ms | 18416.054962 | 17828.586114 |
| total wall ms | 19055.161428 | 18452.035174 |
| prefill tok/s | 37.834213 | 38.064111 |
| decode tok/s | 81.417793 | 81.410307 |
| verified | `true` | `true` |

## 判断

- 同一layer内のrequest slot間でAQ4 payload bufferを共有する最初の境界は通った。
- final top1 tokenは前回のmanifest smokeと一致した。
- `layer_load_ms` は約 `587.468848 ms` 短縮したが、まだ大部分のload timeは残っている。
- これは完全なshared resident weightsではない。RMSNorm/aux passthrough buffers、AQ4 `scale_values_buffer`、workspace、Conv1d history、recurrent state、paged KV cacheはまだslotごとに持っている。
- `prefill_real_batch=false` / `decode_real_batch=false` のままなので、SQ/vLLM throughput比較には使わない。

## 次の行動

1. `scale_values_buffer` とpassthrough weight buffersをslot間で共有できるようにする。
2. workspace/state/cacheを分離したまま、weight-only resident bundleをlayerごとに1つへ寄せる。
3. その後、request-batch stepをreal batch executorへ置き換えてfull package throughput rowを作る。
